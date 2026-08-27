"""
load_test_posts.py — publish sample posts to the running mock server via
its real /admin/publish endpoint, so mock_server.py generates correctly
typed id/account/created_at fields itself instead of hand-matching its
internal schema.

Usage:
    python mock_server.py                             (in one terminal)
    python load_test_posts.py data/agent_test_posts.json   (in another)
"""
import json
import sys
import time

import requests

BASE_URL = "http://127.0.0.1:8000"


def main(path: str) -> None:
    with open(path, "r") as f:
        posts = json.load(f)

    for i, post in enumerate(posts, start=1):
        resp = requests.post(f"{BASE_URL}/admin/publish", json={"content": post["content"]})
        resp.raise_for_status()
        published = resp.json()
        print(f"[{i}/{len(posts)}] published id={published['id']}: {published['content']!r}")
        time.sleep(0.2)  # small pause so timestamps are distinguishable / it reads like a live feed


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "posts.json")