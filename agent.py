"""
agent.py — Truth Social stock-mention alert agent.
 
Ties together:
  - sources.PostSource
  - detection
  - alert_channels.AlertChannel

Run against the local mock server for the demo:
    python mock_server.py
    python agent.py --poll-interval 5 --alert-channel console
 
State (dedup cursor + which posts have been alerted) is persisted to
--state-file so a restart doesn't reprocess or drop posts.
"""
 
import argparse
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
 
import requests
 
from sources import MockSource, PostSource
from alert_channels import (
    Alert, 
    AlertChannel, 
    ConsoleAlertChannel, 
    DiscordWebhookAlertChannel, 
    send_with_retry
)
 
logger = logging.getLogger(__name__)
 
 
# --------------------------------------------------------------------------
# Detection (Part 2 hook)
# --------------------------------------------------------------------------
 
@dataclass
class DetectionResult:
    is_stock_related: bool
    tickers: list[str] = field(default_factory=list)
    companies: list[str] = field(default_factory=list)
    method: str = "unknown"
 
 
def _placeholder_detector(text: str) -> DetectionResult:
    """
    TEMPORARY stand-in for Part 2's real detector.
 
    Only catches an explicit $TICKER pattern. No company-name resolution,
    no false-positive handling (ticker ambiguity, "apple" the fruit, etc).
    This exists so Part 1 + Part 3 are demoable end-to-end before Part 2
    lands. Replace with a call into the real detector module, e.g.:
 
        from detector import detect
        detection = detect(text)
    """
    tickers = re.findall(r"\$([A-Za-z]{1,5})\b", text)
    tickers = [t.upper() for t in tickers]
    return DetectionResult(
        is_stock_related=bool(tickers),
        tickers=tickers,
        companies=[],
        method="placeholder-regex",
    )
 
 
# --------------------------------------------------------------------------
# State persistence (deduplication across restarts)
# --------------------------------------------------------------------------
 
@dataclass
class AgentState:
    last_id: int | None = None
    alerted_ids: list[int] = field(default_factory=list)
    pending_alerts: list[dict] = field(default_factory=list)
    last_successful_poll_at: str | None = None
    last_new_post_at: str | None = None
 
    @classmethod
    def load(cls, path: Path) -> "AgentState":
        if not path.exists():
            return cls()
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)
 
    def save(self, path: Path) -> None:
        # atomic write so a crash mid-write can't corrupt state
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with open(tmp_path, "w") as f:
            json.dump(self.__dict__, f, indent=2)
        os.replace(tmp_path, path)
 
 
# --------------------------------------------------------------------------
# Core agent
# --------------------------------------------------------------------------
 
class Agent:
    def __init__(
        self,
        source: PostSource,
        alert_channel: AlertChannel,
        state_path: Path,
        poll_interval: float,
        heartbeat_threshold: timedelta,
        flush_interval: float | None = None,
        quiet_threshold: timedelta = timedelta(hours=24),
    ):
        self.source = source
        self.alert_channel = alert_channel
        self.state_path = state_path
        self.poll_interval = poll_interval
        self.flush_interval = flush_interval if flush_interval is not None else poll_interval
        self.heartbeat_threshold = heartbeat_threshold
        self.quiet_threshold = quiet_threshold
        self.state = AgentState.load(state_path)
        self._consecutive_failures = 0
 
    # -- main loop ---------------------------------------------------------
 
    def run(self) -> None:
        logger.info(
            "Agent starting. poll_interval=%ss flush_interval=%ss state_file=%s",
            self.poll_interval, self.flush_interval, self.state_path,
        )
        next_poll = 0.0
        next_flush = 0.0
        try:
            while True:
                now = time.monotonic()
                if now >= next_poll:
                    self.poll_once()
                    self._check_heartbeat()
                    next_poll = now + self.poll_interval
                if now >= next_flush:
                    self._flush_pending_alerts()
                    next_flush = now + self.flush_interval
                time.sleep(min(self.flush_interval, self.poll_interval, 5))
        except KeyboardInterrupt:
            logger.info("Shutdown requested, exiting cleanly.")
            self.state.save(self.state_path)
 
    def poll_once(self) -> None:
        try:
            posts = self.source.get_new_posts()
        except requests.exceptions.ConnectionError as e:
            self._on_poll_failure(f"source unreachable: {e}")
            return
        except requests.exceptions.RequestException as e:
            # covers HTTP errors (raise_for_status), timeouts, rate limiting, etc.
            self._on_poll_failure(f"request error: {e}")
            return
        except ValueError as e:
            # malformed JSON
            self._on_poll_failure(f"malformed response: {e}")
            return
 
        self._consecutive_failures = 0
        now_iso = datetime.now(timezone.utc).isoformat()
        self.state.last_successful_poll_at = now_iso
        self.state.save(self.state_path)  # persist the heartbeat timestamp promptly
 
        new_posts = self._filter_and_sort_new(posts)
        logger.info("Poll complete: %d new post(s) found", len(new_posts))
        for post in new_posts:
            self._process_post(post)
            self.state.save(self.state_path)  # incremental save per post, not just at the end
 
    def _filter_and_sort_new(self, posts: list[dict]) -> list[dict]:
        """Dedup lives here. Filters to posts newer than our persisted cursor
        (state.last_id), oldest-first so alerts fire in publish order."""
        valid = []
        required_fields = {"id", "content", "created_at"}
        for p in posts:
            if not required_fields.issubset(p.keys()):
                logger.warning("Skipping malformed post (missing fields): %r", p)
                continue
            valid.append(p)
 
        if self.state.last_id is not None:
            valid = [p for p in valid if p["id"] > self.state.last_id]
 
        return sorted(valid, key=lambda p: p["id"])
 
    def _process_post(self, post: dict) -> None:
        """
        Detection and queueing of each post. This function does not send alerts,
        it just flags stock related posts to be be alerted. Delivery occurs
        during flush.
        """
        self.state.last_new_post_at = datetime.now(timezone.utc).isoformat()

        detection = _placeholder_detector(post["content"])  # TODO: swap for Part 2 detector

        already_alerted = post["id"] in self.state.alerted_ids
        already_pending = any(p["post_id"] == post["id"] for p in self.state.pending_alerts)

        if detection.is_stock_related and not already_alerted and not already_pending:
            alert = Alert(
                post_id=post["id"],
                text=post["content"],
                created_at=post["created_at"],
                tickers=detection.tickers,
                companies=detection.companies,
            )
            self.state.pending_alerts.append(asdict(alert))
            logger.info("Post %s queued for alerting (tickers=%s companies=%s)", post["id"], detection.tickers, detection.companies)

        # advance the cursor once detection has run, regardless of delivery —
        # delivery is now the pending queue's problem, not this method's.
        self.state.last_id = max(self.state.last_id or 0, post["id"])

    def _flush_pending_alerts(self) -> None:
        """
        Attempt delivery for the backlog, oldest first. Each entry gets a
        short burst of immediate retries via send_with_retry. Anything that 
        doesn't resolve within that burst stays queued for the next flush.

        State is saved after every individual attempt, not just at the end
        of the batch, so a crash mid-flush can't cause a duplicate send on
        restart.
        """
        if not self.state.pending_alerts:
            return

        still_pending = []
        for entry in self.state.pending_alerts:
            alert = Alert(**entry)

            if send_with_retry(self.alert_channel, alert, max_attempts=2):
                delivered_at = datetime.now(timezone.utc)
                latency = delivered_at - datetime.fromisoformat(alert.created_at)
                logger.info(
                    "Alert delivered for post %s. end-to-end latency=%.2fs",
                    alert.post_id, latency.total_seconds(),
                )
                self.state.alerted_ids.append(alert.post_id)
            else:
                logger.warning("Delivery failed for post %s; will retry next flush cycle.", alert.post_id)
                still_pending.append(entry)

            self.state.save(self.state_path)

        self.state.pending_alerts = still_pending
        self.state.save(self.state_path)
 
    # -- failure / monitoring ----------------------------------------------
 
    def _on_poll_failure(self, reason: str) -> None:
        self._consecutive_failures += 1
        backoff = min(self.poll_interval * (2 ** self._consecutive_failures), 300)      # 5 minute maxiumum
        logger.error("Poll failed (%d consecutive): %s. Backing off %.0fs.", self._consecutive_failures, reason, backoff)
        time.sleep(backoff)
 
    def _check_heartbeat(self) -> None:
        """TODO: wire this into the alert channel too (e.g. a distinct
        '#monitoring' style message) — for now it's log-only.

        Two independent signals, since they mean different things:
        - ingestion heartbeat: have we successfully POLLED recently? If not,
            the ingestion method itself is likely broken (endpoint changed,
            auth expired, mock server down).
        - quiet account: have we seen a NEW post recently, even though
            polling itself keeps succeeding? Not necessarily a problem but 
            valuable information since the likely cause, and subsequent fix, 
            differs from a broken poller.
        """
        now = datetime.now(timezone.utc)

        if self.state.last_successful_poll_at is not None:
            last_poll = datetime.fromisoformat(self.state.last_successful_poll_at)
            if now - last_poll > self.heartbeat_threshold:
                logger.warning(
                    "No successful poll in over %s — ingestion may be silently broken.",
                    self.heartbeat_threshold,
                )

        if self.state.last_new_post_at is not None:
            last_new_post = datetime.fromisoformat(self.state.last_new_post_at)
            if now - last_new_post > self.quiet_threshold:
                logger.info(
                    "No new posts observed in over %s — account may just be quiet, "
                    "but worth a manual sanity check if unexpected.",
                    self.quiet_threshold,
                )
 
 
# --------------------------------------------------------------------------
# CLI entry point
# --------------------------------------------------------------------------
 
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Truth Social stock-mention alert agent")
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--account", default="demo_user",
                help="account to monitor via the mock source. For production, "
                     "TruthSocialSource defaults to account_handle='realDonaldTrump' "
                     "instead — see sources.py.")
    p.add_argument("--poll-interval", type=float, default=10.0, help="seconds between polls")
    p.add_argument("--flush-interval", type=float, default=None,
                    help="seconds between retrying undelivered alerts; defaults to --poll-interval "
                     "if unset. Set lower than --poll-interval when polling is infrequent.")
    p.add_argument("--state-file", default="agent_state.json")
    p.add_argument("--alert-channel", choices=["console", "discord"], default="console")
    p.add_argument("--webhook-url", default=None, help="required if --alert-channel=discord")
    p.add_argument("--heartbeat-hours", type=float, default=1.0, help="warn if no successful polls occur in this many hours")
    p.add_argument("--quiet-hours", type=float, default=6.0, help="warn if no NEW post observed in this many hours")
    p.add_argument("--log-level", default="INFO")
    return p
 
 
def main() -> None:
    args = build_arg_parser().parse_args()
 
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
 
    source = MockSource(base_url=args.base_url, account=args.account)
 
    if args.alert_channel == "discord":
        if not args.webhook_url:
            sys.exit("--webhook-url is required when --alert-channel=discord")
        channel: AlertChannel = DiscordWebhookAlertChannel(args.webhook_url)
    else:
        channel = ConsoleAlertChannel()
 
    agent = Agent(
        source=source,
        alert_channel=channel,
        state_path=Path(args.state_file),
        poll_interval=args.poll_interval,
        flush_interval=args.flush_interval,
        heartbeat_threshold=timedelta(hours=args.heartbeat_hours),
        quiet_threshold=timedelta(hours=args.quiet_hours),
    )
    agent.run()
 
 
if __name__ == "__main__":
    main()
 
