#!/usr/bin/env python3

import os
import sys
import time
import uuid
import logging
from dotenv import load_dotenv
import weaviate
from weaviate.auth import AuthApiKey
import tracemalloc

tracemalloc.start()
# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("FAQSemanticBot")

# Load env variables
ROOT = os.path.abspath(os.path.dirname(__file__))
load_dotenv()
weaviate_url = os.getenv("WEAVIATE_URL")
weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

if not weaviate_url or not weaviate_api_key or not openai_api_key:
    logger.error("Missing required environment variables.")
    sys.exit(1)

COLLECTION = "FAQ"

FAQ_DATA = [
    {
        "question": "How do I reset my password?",
        "answer": "Go to your account settings, click 'Reset Password', and follow the instructions sent to your email.",
        "category": "account"
    },
    {
        "question": "How can I contact support?",
        "answer": "You can contact support via the chat widget or by emailing support@example.com.",
        "category": "support"
    },
    {
        "question": "What payment methods are accepted?",
        "answer": "We accept Visa, Mastercard, PayPal, and Apple Pay.",
        "category": "billing"
    },
    {
        "question": "How do I delete my account?",
        "answer": "Please email support@example.com with your request to delete your account.",
        "category": "account"
    },
    {
        "question": "Can I change my subscription plan?",
        "answer": "Yes, you can change your subscription anytime from the billing page in your dashboard.",
        "category": "billing"
    }
]

def connect() -> weaviate.Client | None:
    import weaviate
    from weaviate.auth import AuthApiKey

    if not weaviate_url.startswith('https://'):
        url = f"https://{weaviate_url}"
    else:
        url = weaviate_url

    try:
        client = weaviate.Client(
            url=url,
            auth_client_secret=AuthApiKey(api_key=weaviate_api_key),
            additional_headers={"X-OpenAI-Api-Key": openai_api_key}
        )
        logger.info(f"Weaviate client base_url: {url}")
        if not client.is_ready():
            logger.error("Weaviate client is not ready.")
            sys.exit(1)
        return client
    except Exception as exc:
        logger.error(f"Error connecting to Weaviate Cloud: {exc}")
        return None


def setup_collection(client):
    # Delete class if it exists
    try:
        if hasattr(client, "schema") and hasattr(client.schema, "delete_class"):
            schema = client.schema.get()
            if any(cls['class'] == COLLECTION for cls in schema.get('classes', [])):
                client.schema.delete_class(COLLECTION)
                time.sleep(1)
        else:
            logger.error("client.schema interface not available. Update Weaviate client.")
            sys.exit(1)
    except Exception as exc:
        logger.warning(f"Could not check/delete class: {exc}")
    # Create class
    class_obj = {
        "class": COLLECTION,
        "description": "Product FAQ entries",
        "vectorizer": "text2vec-openai",
        "moduleConfig": {
            "text2vec-openai": {"model": "text-embedding-3-small", "type": "text"},
            "generative-openai": {"model": "gpt-4o"}
        },
        "properties": [
            {"name": "question", "dataType": ["text"], "description": "FAQ question"},
            {"name": "answer", "dataType": ["text"], "description": "FAQ answer"},
            {"name": "category", "dataType": ["text"], "description": "FAQ category"}
        ]
    }
    try:
        client.schema.create_class(class_obj)
    except Exception as exc:
        logger.warning(f"Failed to create class: {exc}")
    logger.info("Collection created.")

def ingest_faqs(client):
    for faq in FAQ_DATA:
        obj_id = str(uuid.uuid4())
        try:
            client.data_object.create(faq, COLLECTION, uuid=obj_id)
        except Exception as exc:
            logger.warning(f"Failed to ingest: {faq['question']} ({exc})")
    logger.info("All FAQs ingested.")
    time.sleep(2)  # Wait for indexing

def semantic_search(client, user_q):
    q = f"""
    {{
      Get {{
        {COLLECTION}(
          nearText: {{ concepts: [\"{user_q}\"] }}
          limit: 1
        ) {{
          question
          answer
          category
          _additional {{ certainty }}
        }}
      }}
    }}
    """
    resp = client._connection.post("/graphql", {"query": q})
    if resp.status_code == 200:
        data = resp.json()
        faqs = data.get("data", {}).get("Get", {}).get(COLLECTION, [])
        if faqs:
            return faqs[0]
    return None

def generate_answer(
    user_q: str,
    faq: dict,
):
    """Generate a friendly answer using OpenAI directly.

    Params
    ------
    user_q : str
        The user's raw question.
    faq : dict
        The FAQ entry returned by semantic_search.
    """
    import openai
    openai.api_key = openai_api_key
    openai_client = openai.OpenAI()
    models = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]
    prompt = (
        f"A user asked: '{user_q}'.\n"
        f"The closest FAQ question is: '{faq['question']}'.\n"
        f"The official answer is: '{faq['answer']}'.\n\n"
        "Write a concise, friendly answer for the user, referencing the official FAQ answer but in your own words."
    )

    for model in models:
        try:
            response = openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=256,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            logger.error(f"OpenAI generation failed with model {model}: {exc}")
    return None

def main():
    logger.info("Starting FAQ semantic bot demo.")
    tracemalloc.start()
    try:
        client = connect()
        setup_collection(client)
        ingest_faqs(client)
        logger.info("Ready! Ask a question (or type 'exit'):")
        while True:
            user_q = input("You: ").strip()
            if user_q.lower() in {"exit", "quit"}:
                break
            faq = semantic_search(client, user_q)
            if not faq:
                print("Sorry, I couldn't find a relevant FAQ.")
                continue
            print(f"\nClosest FAQ: {faq['question']}\nOfficial Answer: {faq['answer']}")
            gen = generate_answer(user_q, faq)
            if gen:
                print(f"\nBot: {gen}\n")
            else:
                print("\nBot: (Could not generate a custom answer.)\n")
    except Exception as exc:
        logger.error(f"Error: {exc}")
    finally:
        pass
    logger.info("Session ended.")

if __name__ == "__main__":
    main()
