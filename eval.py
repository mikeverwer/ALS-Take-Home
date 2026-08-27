"""
eval.py - Detector evaluation harness for the ALS take-home.

Loads labels.csv (hand-labeled ground truth) and runs detector.detect()
against every post, comparing:
  - rule-only detection      (use_embeddings=False)
  - rule + embedding detection (use_embeddings=True), at the current
    default threshold AND swept across a range of thresholds

Reports, for each mode:
  - Binary "is this post stock-related?" precision/recall/F1
  - Ticker-level precision/recall/F1
      * strict  : predicted tickers vs. primary ground-truth tickers only
      * lenient : predicted tickers vs. (primary tickers + alt_candidates),
                  since several posts are deliberately ambiguous and a
                  reasonable alternate guess shouldn't be scored as wrong
  - A per-post error log for anything binary-misclassified
  - A threshold sweep table for the embedding pass, with a suggested value

Usage (from an activated venv with sentence-transformers already able to
resolve its model, i.e. previously downloaded / HF_HUB_OFFLINE not blocking
a cached model):

    python eval.py
    python eval.py --thresholds 0.20 0.60 0.025
    python eval.py --labels path\to\labels.csv

If the embedding model can't be loaded (no network / not cached yet), the
rule-only report still runs and prints on its own; the embedding sections
are skipped with a clear message instead of crashing.
"""
import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from detector import detect


@dataclass
class LabeledPost:
    post_id: int
    content: str
    is_stock_related: bool
    tickers: set[str]
    alt_candidates: set[str]
    notes: str

    @property
    def accepted(self) -> set[str]:
        return self.tickers | self.alt_candidates


def load_labels(path: Path) -> list[LabeledPost]:
    posts = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tickers = {t.strip().upper() for t in row["tickers"].split("|") if t.strip()}
            alt = {t.strip().upper() for t in row["alt_candidates"].split("|") if t.strip()}
            posts.append(LabeledPost(
                post_id=int(row["post_id"]),
                content=row["content"],
                is_stock_related=row["is_stock_related"].strip() == "1",
                tickers=tickers,
                alt_candidates=alt,
                notes=row["notes"],
            ))
    return posts


def prf1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def evaluate(posts: list[LabeledPost], use_embeddings: bool, threshold: float = 0.35) -> dict:
    bin_tp = bin_fp = bin_fn = bin_tn = 0
    strict_tp = strict_fp = strict_fn = 0
    lenient_tp = lenient_fp = lenient_fn = 0
    errors = []

    for post in posts:
        kwargs = {"use_embeddings": use_embeddings}
        if use_embeddings:
            kwargs["threshold"] = threshold
        result = detect(post.content, **kwargs)
        pred_related = result.is_stock_related
        pred_tickers = set(result.tickers)

        # --- binary: is this post stock-related at all? ---
        if pred_related and post.is_stock_related:
            bin_tp += 1
        elif pred_related and not post.is_stock_related:
            bin_fp += 1
        elif not pred_related and post.is_stock_related:
            bin_fn += 1
        else:
            bin_tn += 1

        if pred_related != post.is_stock_related:
            errors.append({
                "post_id": post.post_id,
                "content": post.content,
                "type": "false_positive" if pred_related else "false_negative",
                "predicted_tickers": sorted(pred_tickers),
                "ground_truth_tickers": sorted(post.tickers),
                "notes": post.notes,
            })

        # --- ticker-level, only meaningful where ground truth says stock-related ---
        if post.is_stock_related:
            strict_tp += len(pred_tickers & post.tickers)
            strict_fp += len(pred_tickers - post.tickers)
            strict_fn += len(post.tickers - pred_tickers)

            accepted = post.accepted
            lenient_tp += len(pred_tickers & accepted)
            lenient_fp += len(pred_tickers - accepted)
            lenient_fn += len(post.tickers - pred_tickers)
        elif pred_tickers:
            # ground truth says NOT stock-related but we predicted ticker(s) anyway
            strict_fp += len(pred_tickers)
            lenient_fp += len(pred_tickers)

    binary = prf1(bin_tp, bin_fp, bin_fn)
    strict = prf1(strict_tp, strict_fp, strict_fn)
    lenient = prf1(lenient_tp, lenient_fp, lenient_fn)
    return {
        "binary": {"tp": bin_tp, "fp": bin_fp, "fn": bin_fn, "tn": bin_tn,
                   "precision": binary[0], "recall": binary[1], "f1": binary[2]},
        "ticker_strict": {"tp": strict_tp, "fp": strict_fp, "fn": strict_fn,
                           "precision": strict[0], "recall": strict[1], "f1": strict[2]},
        "ticker_lenient": {"tp": lenient_tp, "fp": lenient_fp, "fn": lenient_fn,
                            "precision": lenient[0], "recall": lenient[1], "f1": lenient[2]},
        "errors": errors,
    }


def print_report(name: str, result: dict, show_errors: bool = True) -> None:
    print(f"\n=== {name} ===")
    for key, label in (("binary", "binary (is-stock-related)"),
                        ("ticker_strict", "ticker (strict)"),
                        ("ticker_lenient", "ticker (lenient, incl. alt_candidates)")):
        m = result[key]
        print(f"  {label:40s} P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}  "
              f"(tp={m['tp']} fp={m['fp']} fn={m['fn']})")
    if show_errors and result["errors"]:
        print(f"  {len(result['errors'])} binary misclassification(s):")
        for e in result["errors"]:
            print(f"    [{e['type']}] #{e['post_id']}: {e['content']!r}")
            print(f"        predicted={e['predicted_tickers']}  truth={e['ground_truth_tickers']}  | {e['notes']}")


def sweep_thresholds(posts: list[LabeledPost], lo: float, hi: float, step: float):
    print(f"\n=== Threshold sweep ({lo}-{hi}, step {step}) ===")
    print(f"  {'thresh':>7s}  {'bin_F1':>7s}  {'strict_F1':>10s}  {'lenient_F1':>11s}")
    best = None
    t = lo
    while t <= hi + 1e-9:
        result = evaluate(posts, use_embeddings=True, threshold=t)
        combined_score = (result["binary"]["f1"] + result["ticker_lenient"]["f1"]) / 2
        print(f"  {t:7.3f}  {result['binary']['f1']:7.3f}  {result['ticker_strict']['f1']:10.3f}  "
              f"{result['ticker_lenient']['f1']:11.3f}")
        if best is None or combined_score > best[1]:
            best = (t, combined_score, result)
        t += step
    print(f"\n  Best threshold by (binary_F1 + lenient_ticker_F1)/2: {best[0]:.3f}  (score={best[1]:.3f})")
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default="labels.csv")
    parser.add_argument("--thresholds", nargs=3, type=float, default=[0.20, 0.60, 0.025],
                         metavar=("LO", "HI", "STEP"))
    args = parser.parse_args()

    posts = load_labels(Path(args.labels))
    n_pos = sum(p.is_stock_related for p in posts)
    print(f"Loaded {len(posts)} labeled posts ({n_pos} stock-related, {len(posts) - n_pos} not).")

    rule_only = evaluate(posts, use_embeddings=False)
    print_report("Rule-based only", rule_only)

    try:
        combined_default = evaluate(posts, use_embeddings=True, threshold=0.35)
        print_report("Rule + Embedding (threshold=0.35, current default)", combined_default)
        best = sweep_thresholds(posts, *args.thresholds)
        print("\n=== Best-threshold report ===")
        print_report(f"Rule + Embedding (threshold={best[0]:.3f})", best[2])
    except Exception as e:
        print(f"\n[embedding pass unavailable: {e!r}]")
        print("Rule-only results above still stand; run again once the "
              "sentence-transformers model can be loaded (network access or "
              "a locally cached model) to get the embedding comparison and "
              "threshold sweep.")


if __name__ == "__main__":
    main()