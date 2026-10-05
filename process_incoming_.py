"""
Hybrid RAG teaching assistant.

- Embeddings: always local (bge-m3 via Ollama) -> works offline.
- Answering:  Groq when online (fast), automatic fallback to local Ollama
              when there is no internet, no key, a timeout, or a rate limit.

Usage:
    python process_incoming_hybrid.py              # auto (Groq, fallback to Ollama)
    python process_incoming_hybrid.py --offline    # force local only
    python process_incoming_hybrid.py --online     # force Groq only (no fallback)
"""
import os
import sys
import numpy as np
import joblib
import requests
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3")
EMBED_MODEL = "bge-m3"
TOP_K = 5

MODE = "auto"
if "--offline" in sys.argv:
    MODE = "offline"
elif "--online" in sys.argv:
    MODE = "online"


# ---------- helpers ----------
def fmt_time(seconds):
    seconds = int(seconds)
    return f"{seconds // 60}:{seconds % 60:02d}"


def create_embedding(text_list):
    try:
        r = requests.post(f"{OLLAMA_URL}/api/embed",
                          json={"model": EMBED_MODEL, "input": text_list},
                          timeout=60)
        r.raise_for_status()
        return r.json()["embeddings"]
    except requests.exceptions.ConnectionError:
        sys.exit("Ollama is not running. Start it (open the Ollama app or run "
                 "'ollama serve') and make sure bge-m3 is pulled: ollama pull bge-m3")


# ---------- LLM backends ----------
def ask_groq(prompt):
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set")
    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}",
                 "Content-Type": "application/json"},
        json={"model": GROQ_MODEL,
              "messages": [{"role": "user", "content": prompt}]},
        timeout=20,
    )
    r.raise_for_status()  # raises on 401/429/5xx so we can fall back
    return r.json()["choices"][0]["message"]["content"]


def ask_ollama(prompt):
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": OLLAMA_MODEL,
                  "prompt": prompt,
                  "stream": False,
                  "think": False,          # skip qwen3's long reasoning
                  "keep_alive": "30m",     # keep model loaded between questions
                  "options": {"num_predict": 400}},
            timeout=300,
        )
        r.raise_for_status()
        return r.json()["response"]
    except requests.exceptions.ConnectionError:
        sys.exit("Ollama is not running, so there is no offline fallback. "
                 f"Start Ollama and run: ollama pull {OLLAMA_MODEL}")


def inference(prompt):
    """Returns (answer, backend_used)."""
    if MODE != "offline":
        try:
            return ask_groq(prompt), f"Groq ({GROQ_MODEL})"
        except (requests.RequestException, RuntimeError, KeyError) as e:
            if MODE == "online":
                raise
            print(f"[Groq unavailable: {type(e).__name__}] -> using local {OLLAMA_MODEL}...")
    return ask_ollama(prompt), f"Ollama ({OLLAMA_MODEL})"


# ---------- RAG ----------
df = joblib.load("embeddings.joblib")
emb_matrix = np.vstack(df["embeddings"])


def build_prompt(query, top_df):
    chunks = "\n".join(
        f"- Video {row['video no.']} ({row['name']}), "
        f"{fmt_time(row['start'])}-{fmt_time(row['end'])}: {row['text'].strip()}"
        for _, row in top_df.iterrows()
    )
    return f"""You are a teaching assistant for a web development course (Sigma web development course by CodeWithHarry, used with his consent).
Below are subtitle chunks from the course videos, with video number, title, and timestamps (mm:ss):

{chunks}

User question: "{query}"

Rules:
- Tell the user which video and timestamp to watch, and what is taught there.
- Only say a topic is "taught" if the chunks actually explain it. If it is only mentioned in passing, or not present, say so clearly instead of guessing.
- If the question is unrelated to the course, say you can only answer course-related questions.
- Keep the answer short and clear."""


def answer(query):
    q_emb = create_embedding([query])[0]
    sims = cosine_similarity(emb_matrix, [q_emb]).flatten()
    top_idx = sims.argsort()[::-1][:TOP_K]
    top_df = df.iloc[top_idx]
    return inference(build_prompt(query, top_df))


if __name__ == "__main__":
    print(f"Mode: {MODE}. Type 'exit' to quit.\n")
    while True:
        q = input("Ask a question: ").strip()
        if q.lower() in {"exit", "quit", "q"}:
            break
        if not q:
            continue
        text, backend = answer(q)
        print(f"\n{text}\n\n(answered by {backend})\n")
        with open("response.txt", "w", encoding="utf-8") as f:
            f.write(text)