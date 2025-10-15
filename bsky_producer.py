import os, json, time, re, asyncio, websockets
from datetime import datetime, timezone
from kafka import KafkaProducer

# --- CONFIG ---
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:19092")  # host-mapped port
TOPIC = os.getenv("TOPIC", "posts_scored")
# Jetstream endpoint (Bluesky firehose). Use the one that worked for you earlier.
JETSTREAM_URL = os.getenv("JETSTREAM_URL", "wss://jetstream1.us-east.bsky.network/subscribe")

# --- Heuristic scorer (replace with real model when ready) ---
SENSATIONAL = {
    "secreto","bomba","escándalo","impactante","milagro","cura","inmediato","gratis",
    "vacuna secreta","en 2 días","100% garantizado","click aquí","imperdible","viral"
}
EXCLAM_RE = re.compile(r"!{2,}")
URL_RE = re.compile(r"https?://\S+")

def score_fake(text: str) -> float:
    t = text.lower()
    score = 0.0
    # features
    excls = 1.0 if EXCLAM_RE.search(t) else 0.0
    urls  = 1.0 if URL_RE.search(t) else 0.0
    sens  = sum(1 for w in SENSATIONAL if w in t)
    length = len(t)

    # naive linear combo (clip 0-1)
    score = 0.15*excls + 0.15*urls + 0.1*min(sens,3) + (0.1 if length<80 else 0.0)
    return max(0.0, min(1.0, score))

# --- Kafka ---
producer = KafkaProducer(
    bootstrap_servers=KAFKA_BOOTSTRAP.split(","),
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    linger_ms=50,
    acks="1",
)

async def run():
    print(f"[producer] connecting to {JETSTREAM_URL}")
    async with websockets.connect(JETSTREAM_URL, open_timeout=20) as ws:
        print("[producer] connected")
        async for raw in ws:
            try:
                evt = json.loads(raw)
                # Adjust to the actual payload you observed earlier:
                # Many Jetstream examples carry commit + ops + record (post).
                rec = (evt.get("commit") or {}).get("record") or {}
                text = rec.get("text")
                if not text:
                    continue

                post = {
                    "id": evt.get("commit", {}).get("cid") or evt.get("did") or f"bsky-{int(time.time()*1000)}",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "text": text,
                    "author": evt.get("did") or "unknown",
                    "followers": 0,
                    "retweets": 0,
                    "language": rec.get("lang") or "und",
                    "source": "bluesky",
                }
                post["prob_fake"] = round(score_fake(post["text"]), 3)

                producer.send(TOPIC, post)
            except Exception as e:
                print("[producer] error:", e)

if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass
