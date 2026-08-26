import requests

with open(".env/discord-webhook") as f:
    webhook_url = f.readline
requests.post(webhook_url, json={"content": f"🚨 Stock mention detected: {ticker}\n{post_text}\n{timestamp}"})