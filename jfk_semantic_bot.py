"""Semantic QA bot over JFK Declassified Files.

Run with:
    uv run jfk_semantic_bot.py

Ask: "Who killed JFK?" etc.
"""

from __future__ import annotations

import os
import sys
import time
import uuid
import logging
import tracemalloc
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from dotenv import load_dotenv

# Load env vars from project root .env
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

weaviate_url = os.getenv("WEAVIATE_URL")
weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

if not all([weaviate_url, weaviate_api_key, openai_api_key]):
    logger.error("Missing required environment variables.")
    sys.exit(1)

COLLECTION = "JFKDocs"
TXT_FILE = Path(__file__).with_name("JFK-Files-Part-2.txt")
CHARS_PER_CHUNK = 4000  # ~ 1024 tokens (approx)


def connect():
    import weaviate
    from weaviate.auth import AuthApiKey

    url = weaviate_url if weaviate_url.startswith("https://") else f"https://{weaviate_url}"

    client = weaviate.Client(
        url=url,
        auth_client_secret=AuthApiKey(api_key=weaviate_api_key),
        additional_headers={"X-OpenAI-Api-Key": openai_api_key},
    )
    logger.info("Weaviate client base_url: %s", url)
    if not client.is_ready():
        logger.error("Weaviate client is not ready.")
        sys.exit(1)
    return client


def setup_collection(client):
    # delete existing
    try:
        schema = client.schema.get()
        if any(cls["class"] == COLLECTION for cls in schema.get("classes", [])):
            client.schema.delete_class(COLLECTION)
            time.sleep(1)
    except Exception as exc:
        logger.warning("Could not reset collection: %s", exc)

    class_obj: Dict[str, Any] = {
        "class": COLLECTION,
        "description": "Chunks of JFK declassified files",
        "vectorizer": "text2vec-openai",
        "moduleConfig": {
            "text2vec-openai": {"model": "text-embedding-3-small", "type": "text"},
            "generative-openai": {"model": "gpt-4o"},
        },
        "properties": [
            {"name": "text", "dataType": ["text"], "description": "Content chunk"},
            {"name": "chunk_id", "dataType": ["text"], "description": "Chunk identifier"},
        ],
    }
    client.schema.create_class(class_obj)
    logger.info("Collection created.")


def _read_chunks() -> List[str]:
    if not TXT_FILE.exists():
        logger.error("Text file not found: %s", TXT_FILE)
        sys.exit(1)

    chunks: List[str] = []
    curr: List[str] = []
    curr_len = 0
    with TXT_FILE.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if line.strip() == "":
                # treat paragraph break
                line = "\n"
            if curr_len + len(line) > CHARS_PER_CHUNK:
                chunks.append("".join(curr).strip())
                curr = [line]
                curr_len = len(line)
            else:
                curr.append(line)
                curr_len += len(line)
        if curr:
            chunks.append("".join(curr).strip())
    logger.info("Prepared %d chunks for ingestion.", len(chunks))
    return chunks


def _ingest_one(client, idx: int, chunk: str):
    obj = {"text": chunk, "chunk_id": f"part2_{idx}"}
    client.data_object.create(obj, COLLECTION, uuid=str(uuid.uuid4()))


def ingest_docs(client):
    chunks = _read_chunks()
    workers = min(32, os.cpu_count())
    logger.info("Ingesting %d chunks with %d workers...", len(chunks), workers)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_ingest_one, client, idx, chunk): idx for idx, chunk in enumerate(chunks)
        }
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Ingesting"):
            pass
    logger.info("All chunks ingested.")
    time.sleep(3)


def semantic_search(client, query: str) -> Dict[str, Any] | None:
    try:
        res = (
            client.query.get(COLLECTION, ["text", "chunk_id"])
            .with_near_text({"concepts": [query]})
            .with_limit(1)
            .do()
        )
        docs = res.get("data", {}).get("Get", {}).get(COLLECTION, [])
        if docs:
            return docs[0]
    except Exception as exc:
        logger.error("Search failed: %s", exc)
    return None


def generate_answer(user_q: str, doc: Dict[str, Any]) -> str | None:
    import openai

    openai_client = openai.OpenAI()
    prompt = (
        f"A user asked: '{user_q}'.\n"
        "Here is a passage from declassified JFK documents that might be relevant:\n"\
        f"---\n{doc['text']}\n---\n"\
        "Based only on the information provided in the passage, answer the user's question in a concise, factual way."
    )

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=512,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        logger.error("OpenAI generation failed: %s", exc)
        return None


def main():
    logger.info("Starting JFK semantic bot demo.")
    tracemalloc.start()
    client = connect()
    setup_collection(client)
    ingest_docs(client)

    logger.info("Ready! Ask a question (or type 'exit'):")
    while True:
        try:
            user_q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user_q.lower() in {"exit", "quit"}:
            break
        doc = semantic_search(client, user_q)
        if not doc:
            print("Sorry, I couldn't find relevant info.")
            continue
        print(f"\nClosest passage (chunk {doc['chunk_id']}):\n{doc['text'][:300]}...\n")
        answer = generate_answer(user_q, doc)
        if answer:
            print(f"\nBot: {answer}\n")
        else:
            print("\nBot: (Could not generate an answer.)\n")

    logger.info("Session ended.")


if __name__ == "__main__":
    main()
