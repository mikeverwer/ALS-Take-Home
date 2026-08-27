"""
test_alert_channels.py - unit tests for send_with_retry's retry/backoff
behavior in alert_channels.py.

All tests use a FakeChannel instead of ConsoleAlertChannel/Discord, and
monkeypatch time.sleep to a no-op - so these run instantly and offline,
with no real waiting and no real webhook. That's the "recorded fixture"
equivalent for this module: a scripted sequence of send() outcomes
standing in for what a real channel would do over several attempts.

NOTE ON SCOPE: idempotency itself (never alerting the same post twice)
is explicitly the *caller's* job per this module's own docstring -
Agent tracks alerted_ids/pending_alerts and persists them across
restarts. That behavior lives in agent.py, which isn't in the project
directory as of this test file's writing - see test_agent.py (and the
"missing agent.py" note) for that half.
"""
from dataclasses import replace

import pytest

from alert_channels import Alert, AlertChannel, send_with_retry


SAMPLE_ALERT = Alert(
    post_id=1,
    text="Just bought more $TSLA",
    created_at="2026-01-01T00:00:00Z",
    tickers=["TSLA"],
    companies=["Tesla"],
)


class FakeChannel(AlertChannel):
    """Scripted channel: send() pops one outcome off a list each call."""
    def __init__(self, outcomes, retry_after=None):
        self.outcomes = list(outcomes)
        self.retry_after = retry_after
        self.calls = 0

    def send(self, alert: Alert) -> bool:
        self.calls += 1
        return self.outcomes.pop(0)


@pytest.fixture(autouse=True)
def no_real_sleep(monkeypatch):
    # Every test in this file runs instantly - no real backoff delay.
    monkeypatch.setattr("alert_channels.time.sleep", lambda seconds: None)


def test_succeeds_on_first_attempt_without_retrying():
    channel = FakeChannel([True])
    assert send_with_retry(channel, SAMPLE_ALERT, max_attempts=3) is True
    assert channel.calls == 1


def test_retries_then_succeeds():
    channel = FakeChannel([False, False, True])
    assert send_with_retry(channel, SAMPLE_ALERT, max_attempts=3) is True
    assert channel.calls == 3


def test_gives_up_after_max_attempts():
    channel = FakeChannel([False, False, False])
    assert send_with_retry(channel, SAMPLE_ALERT, max_attempts=3) is False
    assert channel.calls == 3  # never exceeds max_attempts


def test_stops_retrying_as_soon_as_it_succeeds():
    # if it retried even one extra time after success, this would raise
    # IndexError from popping an empty list.
    channel = FakeChannel([False, True])
    assert send_with_retry(channel, SAMPLE_ALERT, max_attempts=5) is True
    assert channel.calls == 2


def test_respects_server_specified_retry_after(monkeypatch):
    sleep_calls = []
    monkeypatch.setattr("alert_channels.time.sleep", lambda seconds: sleep_calls.append(seconds))

    channel = FakeChannel([False, True], retry_after=30.0)
    assert send_with_retry(channel, SAMPLE_ALERT, max_attempts=3, max_wait=60.0) is True
    assert sleep_calls == [30.0]


def test_caps_retry_after_at_max_wait(monkeypatch):
    sleep_calls = []
    monkeypatch.setattr("alert_channels.time.sleep", lambda seconds: sleep_calls.append(seconds))

    # Discord says wait 120s, but we cap backoff at 60s.
    channel = FakeChannel([False, True], retry_after=120.0)
    assert send_with_retry(channel, SAMPLE_ALERT, max_attempts=3, max_wait=60.0) is True
    assert sleep_calls == [60.0]