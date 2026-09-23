"""Label 2214 hermes session messages using the GLM-4.7-Flash classifier on docker-ssd.
Outputs traces in the same JSONL format as the router-proxy, so fit_surrogate.py can consume them.
"""
import json, time, re, sys
import httpx

# Router-proxy classifier config
CATEGORIES = ["chat", "code", "devops", "research", "homeassistant"]
SYSTEM_PROMPT = """Classify the user's message into exactly one of these categories: {categories}

Examples:
"Hello" → chat
"Write a Python function" → code
"Deploy to Proxmox" → devops
"What is quantum computing?" → research
"Thanks!" → chat
"Debug nginx 502" → code
"Create HA automation" → homeassistant

Reply with ONLY the category name, nothing else.

"{message}" →
"""

BASE_URL = "https://ollama.com/v1"
MODEL = "glm-5.3-flash"
API_KEY = "d61b4e981d87488288b8c69aae71a39b.6PudgrhjZivTNFpCoLDvFyPb"

def classify(msg: str) -> str:
    prompt = SYSTEM_PROMPT.format(categories=", ".join(CATEGORIES), message=msg)
    try:
        resp = httpx.post(
            f"{BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": "You are a classifier. Reply with ONLY the category name, nothing else."},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 256,
                "temperature": 0.0,
                "stream": False,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=120.0,
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip().lower()
        # Extract just the category name
        for cat in CATEGORIES:
            if cat in text:
                return cat
        return "chat"  # default
    except Exception as e:
        return f"ERR:{type(e).__name__}"

# Load messages
with open("/root/router-proxy/traces/hermes_session_msgs.json") as f:
    msgs = json.load(f)

print(f"Labeling {len(msgs)} messages with {MODEL} on {BASE_URL}...", flush=True)

# Warmup
classify("hello")

labeled = []
errors = 0
t0 = time.perf_counter()

for i, msg in enumerate(msgs):
    cat = classify(msg)
    if cat.startswith("ERR:"):
        errors += 1
        if errors > 50:
            print(f"Too many errors ({errors}), stopping", flush=True)
            break
        time.sleep(1)
        continue

    # Write in router-proxy trace format
    entry = {
        "ts": f"2026-09-21T00:00:00.{i:06d}Z",
        "event": "classify",
        "session_key": f"hermes_session_{i}",
        "user_message_preview": msg,
        "classifier_result": cat,
        "classifier_raw": cat,
        "latency_ms": 0,
        "tier": cat,
        "model": "hermes-session-labeler",
        "is_first": True,
    }
    labeled.append(entry)

    if (i + 1) % 100 == 0:
        elapsed = time.perf_counter() - t0
        rate = (i + 1) / elapsed
        eta = (len(msgs) - i - 1) / rate
        print(f"  {i+1}/{len(msgs)} labeled ({rate:.1f}/s, ETA {eta:.0f}s, errors={errors})", flush=True)

# Write traces
out_path = "/root/router-proxy/traces/router-trace-hermes-sessions.jsonl"
with open(out_path, "w") as f:
    for entry in labeled:
        f.write(json.dumps(entry) + "\n")

from collections import Counter
dist = Counter(e["classifier_result"] for e in labeled)
print(f"\nDone: {len(labeled)} labeled, {errors} errors", flush=True)
print(f"Distribution: {dict(dist)}", flush=True)
print(f"Saved to: {out_path}", flush=True)