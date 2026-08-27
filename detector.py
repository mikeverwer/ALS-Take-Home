"""
detector.py — Stock-mention detection for the Truth Social alert agent.

Two detection passes, combined:

  1. Rule-based
     - $TICKER regex (the $ prefix is itself a strong, unambiguous signal)
     - Gazetteer alias matching for company names / common short-hands
       ("Truth Social" -> DJT, "Boeing" -> BA, etc.)

  2. Embedding-based entity linking
     - Each gazetteer entry has a short natural-language description.
     - We embed the post text and every description with a small
       sentence-transformer and flag a match when cosine similarity
       clears a threshold.
     - This is what catches purely colloquial references that never
       mention a name or ticker at all, e.g. "the electric car company".

Design decision worth calling out in the write-up: we NEVER treat a bare
uppercase token as a ticker (no $ prefix, not gazetteer-matched). That's
the entire strategy for handling ambiguous short words like "A", "IT",
"ALL" — they simply never get a chance to match. The trade-off is that a
genuine bare-ticker mention for a company outside the gazetteer would be
missed; that's an explicit precision-over-recall choice for a prototype
with a small, curated universe of relevant companies.

Known ambiguity worth flagging in error analysis: because "djt" is listed
as an alias for Trump Media (so posts naming the ticker in prose still
match), a bare mention like "DJT is doing great today" will fire even
when it's plausibly self-referential (his own initials) rather than a
stock reference. We don't have a clean way to disambiguate that from text
alone without more context — flag it, don't silently "fix" it.

Usage:
    from detector import detect
    result = detect(post_text)
"""

import re
from dataclasses import dataclass, field
from functools import lru_cache

# --------------------------------------------------------------------------
# Result type — mirrors agent.py's placeholder DetectionResult so this is a
# drop-in replacement for the `from detector import detect` swap noted there.
# --------------------------------------------------------------------------

@dataclass
class DetectionResult:
    is_stock_related: bool
    tickers: list[str] = field(default_factory=list)
    companies: list[str] = field(default_factory=list)
    method: str = "none"


# --------------------------------------------------------------------------
# Gazetteer — curated, not exhaustive. Scoped to companies plausible for
# this specific account rather than the whole market. That scoping is a
# deliberate trade-off for a 3-day prototype.
# --------------------------------------------------------------------------

@dataclass
class CompanyEntry:
    ticker: str
    canonical_name: str
    aliases: list[str]     # matched case-insensitively, word-boundary
    description: str       # used for embedding similarity — cover colloquial refs here


GAZETTEER: list[CompanyEntry] = [
    CompanyEntry(
        ticker="DJT",
        canonical_name="Trump Media & Technology Group",
        aliases=["trump media", "truth social", "tmtg", "djt"],
        description="Trump Media and Technology Group, the company that owns and operates Truth Social, the social media platform.",
    ),
    CompanyEntry(
        ticker="TSLA",
        canonical_name="Tesla",
        aliases=["tesla"],
        description="Tesla, the electric car and battery company run by Elon Musk.",
    ),
    CompanyEntry(
        ticker="BA",
        canonical_name="Boeing",
        aliases=["boeing"],
        description="Boeing, the aerospace and defense company that builds commercial airplanes.",
    ),
    CompanyEntry(
        ticker="AAPL",
        canonical_name="Apple Inc.",
        aliases=["apple inc", "apple computer"],
        description="Apple Inc, the technology company that makes the iPhone, iPad, and Mac computers.",
    ),
    CompanyEntry(
        ticker="F",
        canonical_name="Ford Motor Company",
        aliases=["ford motor", "ford motors"],
        description="Ford Motor Company, the American car manufacturer.",
    ),
    CompanyEntry(
        ticker="GM",
        canonical_name="General Motors",
        aliases=["general motors"],
        description="General Motors, the American car manufacturer that makes Chevrolet and Cadillac.",
    ),
    CompanyEntry(
        ticker="META",
        canonical_name="Meta Platforms",
        aliases=["facebook", "meta platforms"],
        description="Meta Platforms, the company formerly known as Facebook that owns Facebook and Instagram.",
    ),
    CompanyEntry(
        ticker="GOOGL",
        canonical_name="Alphabet",
        aliases=["google", "alphabet"],
        description="Alphabet, the parent company of Google and YouTube.",
    ),
    CompanyEntry(
        ticker="AMZN",
        canonical_name="Amazon",
        aliases=["amazon"],
        description="Amazon, the e-commerce and cloud computing company founded by Jeff Bezos.",
    ),
    CompanyEntry(
        ticker="NVDA",
        canonical_name="Nvidia",
        aliases=["nvidia"],
        description="Nvidia, the company that designs graphics processing units (GPUs) and AI chips.",
    ),
    CompanyEntry(
        ticker="GME",
        canonical_name="GameStop",
        aliases=["gamestop"],
        description="GameStop, the video game and electronics retail chain that became a meme-stock phenomenon.",
    ),
    CompanyEntry(
        ticker="NFLX",
        canonical_name="Netflix",
        aliases=["netflix"],
        description="Netflix, the video streaming service known for original shows and movies.",
    ),
    CompanyEntry(
        ticker="MSFT",
        canonical_name="Microsoft",
        aliases=["microsoft"],
        description="Microsoft, the technology company that makes Windows, Office, and Xbox.",
    ),
    CompanyEntry(
        ticker="WBD",
        canonical_name="Warner Bros. Discovery",
        aliases=["warner bros discovery", "warner bros. discovery", "hbo max", "warner brothers"],
        description="Warner Bros. Discovery, the media company that owns HBO Max, CNN, and Warner Bros. Studios.",
    ),
    CompanyEntry(
        ticker="DIS",
        canonical_name="The Walt Disney Company",
        aliases=["disney", "walt disney"],
        description="The Walt Disney Company, the entertainment giant behind Disney+, Marvel, Pixar, and theme parks.",
    ),
    CompanyEntry(
        ticker="RIVN",
        canonical_name="Rivian",
        aliases=["rivian"],
        description="Rivian, the electric truck and SUV startup automaker.",
    ),
    CompanyEntry(
        ticker="BYDDF",
        canonical_name="BYD Company",
        aliases=["byd"],
        description="BYD, the Chinese electric vehicle and battery manufacturer.",
    ),
    CompanyEntry(
        ticker="EADSY",
        canonical_name="Airbus",
        aliases=["airbus"],
        description="Airbus, the European aerospace company that builds commercial passenger airplanes, a rival to Boeing.",
    ),
    CompanyEntry(
        ticker="PFE",
        canonical_name="Pfizer",
        aliases=["pfizer"],
        description="Pfizer, the pharmaceutical company known for vaccines and prescription drugs.",
    ),
    CompanyEntry(
        ticker="XOM",
        canonical_name="ExxonMobil",
        aliases=["exxon", "exxonmobil", "exxon mobil"],
        description="ExxonMobil, the multinational oil and gas company.",
    ),
    CompanyEntry(
        ticker="KO",
        canonical_name="The Coca-Cola Company",
        aliases=["coca cola", "coca-cola", "coke"],
        description="The Coca-Cola Company, the beverage company that makes Coke and other soft drinks.",
    ),
    CompanyEntry(
        ticker="AMD",
        canonical_name="Advanced Micro Devices",
        aliases=["amd", "advanced micro devices"],
        description="Advanced Micro Devices (AMD), the semiconductor company that makes CPUs and GPUs, a rival to Nvidia and Intel.",
    ),
    CompanyEntry(
        ticker="TSM",
        canonical_name="Taiwan Semiconductor Manufacturing Company",
        aliases=["tsmc", "taiwan semiconductor"],
        description="Taiwan Semiconductor Manufacturing Company (TSMC), the world's largest contract chip manufacturer.",
    ),
    CompanyEntry(
        ticker="JPM",
        canonical_name="JPMorgan Chase",
        aliases=["jpmorgan", "jp morgan", "chase bank"],
        description="JPMorgan Chase, the largest bank in the United States.",
    ),
    CompanyEntry(
        ticker="INTC",
        canonical_name="Intel",
        aliases=["intel"],
        description="Intel, the semiconductor company best known for making computer processors.",
    ),
    CompanyEntry(
        ticker="WMT",
        canonical_name="Walmart",
        aliases=["walmart", "wal-mart"],
        description="Walmart, the large American retail and grocery chain.",
    ),
    CompanyEntry(
        ticker="CVX",
        canonical_name="Chevron",
        aliases=["chevron"],
        description="Chevron, the multinational oil and gas company.",
    ),
    CompanyEntry(
        ticker="V",
        canonical_name="Visa Inc.",
        # NOTE: "visa" is a strong ambiguity risk (immigration visas vs. the
        # payment company), same shape of problem as Apple the fruit.
        aliases=["visa inc"],
        description="Visa Inc., the payment processing and credit card network company.",
    ),
    CompanyEntry(
        ticker="UNH",
        canonical_name="UnitedHealth Group",
        aliases=["unitedhealth", "united health"],
        description="UnitedHealth Group, the largest health insurance company in the United States.",
    ),
]

_NAME_TO_TICKER = {e.canonical_name: e.ticker for e in GAZETTEER}

TICKER_PATTERN = re.compile(r"\$([A-Za-z]{1,5})\b")

# Tickers safe to match even without a $ prefix or gazetteer alias hit.
# Deliberately restricted to gazetteer tickers that are >=3 characters AND
# don't collide with a common English word/abbreviation in all-caps form -
# this is what keeps this from re-opening the "IT"/"A"/"ALL" problem the
# module docstring warns about above. Single/double-letter tickers (F, V,
# GM, KO, BA) and real-word collisions (META) are deliberately left OUT of
# this set and therefore still require a $ prefix or a gazetteer alias
# match. See eval.py's error analysis for the two known false negatives
# (bare "BA", bare "KO") this trade-off produces.
_COMMON_WORD_COLLISION_TICKERS = {"F", "V", "GM", "KO", "BA", "META"}
SAFE_BARE_TICKERS = {
    e.ticker for e in GAZETTEER
    if len(e.ticker) >= 3 and e.ticker not in _COMMON_WORD_COLLISION_TICKERS
}
_TICKER_TO_ENTRY = {e.ticker: e for e in GAZETTEER}
BARE_TICKER_PATTERN = re.compile(r"\b([A-Z]{3,6})\b")

# Classic false-positive trap the assignment calls out by name.
_FRUIT_CONTEXT_WORDS = {"pie", "eat", "ate", "eating", "fruit", "tree", "orchard", "juice", "cider", "sauce"}

# Immigration-policy vocabulary — the strongest single-word-adjacent signal
# we have for "visa" meaning a travel document rather than the payment
# company. Not exhaustive.
_IMMIGRATION_CONTEXT_WORDS = {
    "immigration", "immigrant", "green card", "visa application", "visa waiver",
    "work permit", "student visa", "tourist visa", "travel visa", "h1b", "h-1b",
    "asylum", "deport", "deportation", "border", "passport", "embassy",
    "citizenship", "naturalization", "visa status", "visa denied", "visa approved",
}

# A modest, non-exhaustive list of country names likely to co-occur with
# "visa" in an immigration context. Nowhere near a full country list —
# this is a coarse heuristic, not a solved problem. Revisit once you can
# see how it performs against labels.csv.
_COUNTRY_NAME_HINTS = {
    "mexico", "china", "india", "canada", "cuba", "haiti", "venezuela",
    "ukraine", "russia", "iran", "afghanistan", "syria", "somalia",
    "guatemala", "honduras", "el salvador", "nicaragua", "colombia",
    "brazil", "philippines", "vietnam", "nigeria", "pakistan", "japan",
    "germany", "france", "uk", "united kingdom", "united states", "usa",
    "america", "spain"
}

# "Chevron" the V-shaped pattern/insignia vs. Chevron the oil company -
# same shape of problem as apple/fruit, surfaced by eval.py against
# labels.csv post #67 ("that shirt has a really nice chevron pattern").
_PATTERN_CONTEXT_WORDS = {
    "pattern", "shirt", "stripe", "stripes", "jacket", "tattoo", "rug",
    "wallpaper", "fabric", "print", "logo", "badge", "insignia", "patch",
}

_GENERIC_RETAIL_CONTEXT_WORDS = {"general store", "farmers market", "corner store", "convenience store"}

def _contains_context_word(lowered_text: str, words: set[str]) -> bool:
    """Word-boundary matching for guard context words."""
    return any(re.search(rf"\b{re.escape(w)}\b", lowered_text) for w in words)


def _is_fruit_context(text: str) -> bool:
    lowered = text.lower()
    return "apple" in lowered and _contains_context_word(lowered, _FRUIT_CONTEXT_WORDS)


def _is_chevron_pattern_context(text: str) -> bool:
    lowered = text.lower()
    return "chevron" in lowered and _contains_context_word(lowered, _PATTERN_CONTEXT_WORDS)


def _is_visa_context(text: str) -> bool:
    """Heuristic guard for 'visa' (immigration paperwork) vs. Visa Inc.
    Harder than the apple/fruit case since there's no single dominant
    clue word. Country names and immigration-policy language are the
    best signals available without a proper NER/context-window pass.
    """
    lowered = text.lower()
    if "visa" not in lowered:
        return False
    if _contains_context_word(lowered, _IMMIGRATION_CONTEXT_WORDS):
        return True
    if _contains_context_word(lowered, _COUNTRY_NAME_HINTS):
        return True
    return False


def _is_generic_retail_context(text: str) -> bool:
    lowered = text.lower()
    return _contains_context_word(lowered, _GENERIC_RETAIL_CONTEXT_WORDS)


# Each entry: (guard function, ticker to suppress, canonical company name to
# suppress). Applied after both detection passes are merged, so a guard
# catches a false positive regardless of whether the rule-based pass or the
# embedding pass produced it. Add new (word, company) ambiguity guards here
# as they turn up in error analysis.
_FALSE_POSITIVE_GUARDS = [
    (_is_fruit_context, "AAPL", "Apple Inc."),
    (_is_visa_context, "V", "Visa Inc."),
    (_is_chevron_pattern_context, "CVX", "Chevron"),
    (_is_generic_retail_context, "WMT", "Walmart"),
]


# --------------------------------------------------------------------------
# Pass 1: rule-based
# --------------------------------------------------------------------------

def _rule_based_pass(text: str) -> tuple[set[str], set[str]]:
    tickers: set[str] = set()
    companies: set[str] = set()

    for match in TICKER_PATTERN.finditer(text):
        tickers.add(match.group(1).upper())

    for match in BARE_TICKER_PATTERN.finditer(text):
        token = match.group(1)
        if token in SAFE_BARE_TICKERS:
            tickers.add(token)
            companies.add(_TICKER_TO_ENTRY[token].canonical_name)

    lowered = text.lower()
    for entry in GAZETTEER:
        for alias in entry.aliases:
            if re.search(rf"\b{re.escape(alias)}\b", lowered):
                tickers.add(entry.ticker)
                companies.add(entry.canonical_name)
                break

    return tickers, companies


# --------------------------------------------------------------------------
# Pass 2: embedding-based entity linking
# Lazy-loaded and cached so the model only loads once per process, not once
# per post — matters for the latency you're asked to measure.
# --------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_embedder():
    from sentence_transformers import SentenceTransformer
    # Explicit cache_folder so the model is guaranteed to persist across runs
    # in one predictable place, instead of relying on the default
    # user-profile cache location (which can behave inconsistently across
    # shells/terminals on Windows).
    return SentenceTransformer("all-MiniLM-L6-v2", cache_folder="./hf_cache")


@lru_cache(maxsize=1)
def _get_gazetteer_embeddings():
    embedder = _get_embedder()
    descriptions = [e.description for e in GAZETTEER]
    return embedder.encode(descriptions, normalize_embeddings=True)


def _embedding_pass(text: str, threshold: float = 0.35) -> set[str]:
    """Returns canonical company names matched purely by semantic
    similarity against the crafted gazetteer, this is what catches 
    colloquial references with no explicit name or ticker.
    e.g. 'the electric car company' -> Tesla.
    """
    embedder = _get_embedder()
    gaz_embeddings = _get_gazetteer_embeddings()
    post_embedding = embedder.encode([text], normalize_embeddings=True)[0]

    sims = gaz_embeddings @ post_embedding  # cosine sim: both sides are unit-normalized
    return {
        entry.canonical_name
        for entry, sim in zip(GAZETTEER, sims)
        if sim >= threshold
    }


# --------------------------------------------------------------------------
# Combined detector
# --------------------------------------------------------------------------

def detect(text: str, use_embeddings: bool = True, threshold: float = 0.35) -> DetectionResult:
    tickers, companies = _rule_based_pass(text)
    rule_hit = bool(tickers or companies)

    new_from_embedding: set[str] = set()
    if use_embeddings:
        embedding_companies = _embedding_pass(text, threshold=threshold)
        new_from_embedding = embedding_companies - companies
        companies.update(new_from_embedding)
        tickers.update(_NAME_TO_TICKER[c] for c in new_from_embedding)

    # Applied last, to the fully merged result — so a guard suppresses a
    # false positive regardless of whether the rule-based pass or the
    # embedding pass is the one that produced it.
    for guard_fn, ticker, company in _FALSE_POSITIVE_GUARDS:
        if guard_fn(text):
            tickers.discard(ticker)
            companies.discard(company)
            new_from_embedding.discard(company)

    if not tickers and not companies:
        method = "none"
    elif new_from_embedding and rule_hit:
        method = "rule+embedding"
    elif new_from_embedding:
        method = "embedding"
    else:
        method = "rule"

    return DetectionResult(
        is_stock_related=bool(tickers or companies),
        tickers=sorted(tickers),
        companies=sorted(companies),
        method=method,
    )


# --------------------------------------------------------------------------
# Quick manual test
# --------------------------------------------------------------------------

if __name__ == "__main__":
    examples = [
        "Just bought more $TSLA, electric cars are the future!",
        "The electric car company is doing incredible things.",
        "I ate an apple for breakfast, delicious.",
        "Apple Inc. stock is way undervalued right now.",
        "IT is a great word, ALL of you should know that.",
        "DJT to the moon!",
        "Truth Social just hit a new record for daily users.",
        "Boeing needs to fix their planes.",
        "I put a lot of stock in loyalty and hard work.",
        "Visa stock is up big today on strong earnings.",
        "My visa application to Canada got denied again.",
    ]
    for text in examples:
        result = detect(text)
        print(f"{text!r}\n  -> {result}\n")
