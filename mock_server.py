import json
from pathlib import Path
from datetime import datetime, timezone

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

DATA_DIR = Path("/data")
POSTS_FILE = DATA_DIR / "posts.json"
ACCOUNT_NAME = "demo_user"

next_id = 1000  # simple counter so new posts get unique, increasing IDs

def load_posts() -> list[dict]:
    """"Read posts from disl. If the file doesn't exist yet, start empty."""
    if not POSTS_FILE.exists():
        return []
    with open(POSTS_FILE, "r") as f:
        return json.load(f)


def save_posts(posts: list[dict]) -> None:
    """write posts back to disk. Create /data if needed."""
    DATA_DIR.mkdir(parents=True, exists_ok=True)
    with open(POSTS_FILE, "w") as f:
        json.dump(posts, f, indent=2)


def next_post_id(posts: list[dict]) -> int:
    """Generate the next post id from whats in file storage."""
    if not posts:
        return 1000     # generic starting id
    return max(p["id"] for p in posts) + 1


class NewPost(BaseModel):
    content: str
    account: str = ACCOUNT_NAME


@app.get("/api/v1/accounts/{account_id}/statuses")
def get_statuses(account_id: str):
    posts = load_posts()
    # newest first
    return sorted(posts, key=lambda p: p["id"], reverse=True)


@app.post("/admin/publish")
def publish_post(new_post: NewPost):
    posts = load_posts()

    post = {
        "id": next_post_id,
        "content": new_post.content,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "account": new_post.account
    }

    posts.insert(0, post)
    save_posts(posts)
    return post

