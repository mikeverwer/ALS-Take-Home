"""
test_agent.py - unit tests for dedup / idempotency / state-persistence
logic in agent.py.

Two things are deliberately kept offline here:
  - No real HTTP: FakeSource stands in for MockSource/TruthSocialSource.
  - No embedding model load: Agent._process_post calls detector.detect()
    with its default use_embeddings=True, which would try to load
    sentence-transformers. Detection logic itself is already covered in
    test_detector.py, so here we monkeypatch agent.detect() to return a
    fixed, known DetectionResult - we're testing dedup/idempotency, not
    re-testing detection.
"""
from datetime import timedelta
from unittest.mock import patch

from agent import Agent, AgentState
from sources import PostSource
from alert_channels import ConsoleAlertChannel
from detector import DetectionResult


STOCK_RELATED = DetectionResult(
    is_stock_related=True, tickers=["TSLA"], companies=["Tesla"], method="rule"
)


class FakeSource(PostSource):
    """In-memory stand-in for MockSource/TruthSocialSource - no HTTP call."""
    def __init__(self, posts):
        self._posts = posts

    def get_new_posts(self):
        return self._posts


def make_agent(tmp_path, state_path=None):
    return Agent(
        source=FakeSource([]),
        alert_channel=ConsoleAlertChannel(),
        state_path=state_path or (tmp_path / "state.json"),
        poll_interval=1.0,
        heartbeat_threshold=timedelta(hours=1),
    )


# --------------------------------------------------------------------------
# State persistence
# --------------------------------------------------------------------------

def test_state_round_trips_through_save_and_load(tmp_path):
    path = tmp_path / "state.json"
    state = AgentState(last_id=5, alerted_ids=[1, 2, 3])
    state.save(path)

    reloaded = AgentState.load(path)
    assert reloaded.last_id == 5
    assert reloaded.alerted_ids == [1, 2, 3]


def test_state_load_handles_missing_file(tmp_path):
    state = AgentState.load(tmp_path / "does_not_exist.json")
    assert state.last_id is None
    assert state.alerted_ids == []


def test_state_load_handles_corrupted_file_without_crashing(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("{not valid json")
    state = AgentState.load(path)  # should degrade gracefully, not raise
    assert state.last_id is None


# --------------------------------------------------------------------------
# Dedup / cursor logic
# --------------------------------------------------------------------------

def test_filter_and_sort_new_dedups_by_last_id(tmp_path):
    agent = make_agent(tmp_path)
    agent.state.last_id = 2
    posts = [
        {"id": 3, "content": "new", "created_at": "2026-01-01T00:00:00Z"},
        {"id": 1, "content": "already seen", "created_at": "2026-01-01T00:00:00Z"},
        {"id": 2, "content": "already seen", "created_at": "2026-01-01T00:00:00Z"},
    ]
    result = agent._filter_and_sort_new(posts)
    assert [p["id"] for p in result] == [3]


def test_filter_and_sort_new_orders_oldest_first(tmp_path):
    agent = make_agent(tmp_path)
    posts = [
        {"id": 5, "content": "c", "created_at": "2026-01-01T00:00:00Z"},
        {"id": 3, "content": "a", "created_at": "2026-01-01T00:00:00Z"},
        {"id": 4, "content": "b", "created_at": "2026-01-01T00:00:00Z"},
    ]
    result = agent._filter_and_sort_new(posts)
    assert [p["id"] for p in result] == [3, 4, 5]


def test_filter_and_sort_new_skips_malformed_posts(tmp_path):
    agent = make_agent(tmp_path)
    posts = [
        {"id": 1, "content": "fine", "created_at": "2026-01-01T00:00:00Z"},
        {"id": 2, "content": "missing timestamp"},                          # no created_at
        {"content": "missing id", "created_at": "2026-01-01T00:00:00Z"},     # no id
    ]
    result = agent._filter_and_sort_new(posts)
    assert [p["id"] for p in result] == [1]


# --------------------------------------------------------------------------
# Idempotency: never queue/send a duplicate alert for the same post
# --------------------------------------------------------------------------

def test_already_alerted_post_is_never_requeued(tmp_path):
    agent = make_agent(tmp_path)
    agent.state.alerted_ids = [42]
    post = {"id": 42, "content": "$TSLA to the moon", "created_at": "2026-01-01T00:00:00Z"}

    with patch("agent.detect", return_value=STOCK_RELATED):
        agent._process_post(post)

    assert agent.state.pending_alerts == []


def test_already_pending_post_is_not_duplicated(tmp_path):
    agent = make_agent(tmp_path)
    agent.state.pending_alerts = [{
        "post_id": 7, "text": "$TSLA update", "created_at": "2026-01-01T00:00:00Z",
        "tickers": ["TSLA"], "companies": ["Tesla"],
    }]
    post = {"id": 7, "content": "$TSLA update", "created_at": "2026-01-01T00:00:00Z"}

    with patch("agent.detect", return_value=STOCK_RELATED):
        agent._process_post(post)

    # still exactly one entry for post 7 - not appended a second time
    assert len(agent.state.pending_alerts) == 1


def test_new_stock_related_post_is_queued_exactly_once(tmp_path):
    agent = make_agent(tmp_path)
    post = {"id": 1, "content": "$TSLA to the moon", "created_at": "2026-01-01T00:00:00Z"}

    with patch("agent.detect", return_value=STOCK_RELATED):
        agent._process_post(post)

    assert len(agent.state.pending_alerts) == 1
    assert agent.state.pending_alerts[0]["post_id"] == 1


def test_alerted_id_persists_across_restart(tmp_path):
    state_path = tmp_path / "state.json"

    # "first run": post 99 already alerted, then the process exits.
    AgentState(last_id=99, alerted_ids=[99]).save(state_path)

    # "restart": a brand-new Agent instance loads that state from disk.
    agent = make_agent(tmp_path, state_path=state_path)
    post = {"id": 99, "content": "$TSLA again", "created_at": "2026-01-01T00:00:00Z"}

    with patch("agent.detect", return_value=STOCK_RELATED):
        agent._process_post(post)

    assert agent.state.pending_alerts == []  # never requeued after "restart"