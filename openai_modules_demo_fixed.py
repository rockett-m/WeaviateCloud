#!/usr/bin/env python3

import os
import sys
import time
import uuid
import logging
import json
from dotenv import load_dotenv
import weaviate
from weaviate.auth import AuthApiKey

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("OpenAIModulesDemo")

# Get the project root directory
ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.append(ROOT)

# Load environment variables from .env file
load_dotenv(os.path.join(ROOT, '.env'))
weaviate_url = os.getenv("WEAVIATE_URL")
weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

if not weaviate_url or not weaviate_api_key or not openai_api_key:
    logger.error("Missing required environment variables: WEAVIATE_URL, WEAVIATE_API_KEY, or OPENAI_API_KEY")
    sys.exit(1)

# Define the collection name for our demo
COLLECTION_NAME = "AIArticles"

def connect_to_weaviate():
    """Connect to Weaviate Cloud instance."""
    # Ensure the URL has https:// prefix
    global weaviate_url
    if not weaviate_url.startswith('https://'):
        weaviate_url = f"https://{weaviate_url}"
    
    logger.info(f"Connecting to Weaviate at {weaviate_url}")
    
    try:
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=weaviate_url,
            auth_credentials=AuthApiKey(api_key=weaviate_api_key),
            headers={
                "X-OpenAI-Api-Key": openai_api_key  # For OpenAI modules
            }
        )
        
        # Check if client is ready
        is_ready = client.is_ready()
        logger.info(f"Connection status: {is_ready}")
        
        if not is_ready:
            logger.error("Weaviate client is not ready")
            return None
        
        return client
    except Exception as e:
        logger.error(f"Error connecting to Weaviate: {e}")
        return None


def check_openai_modules(client):
    """Check if OpenAI modules are available."""
    try:
        meta = client.get_meta()
        
        # Print Weaviate version
        if 'version' in meta:
            logger.info(f"Weaviate version: {meta['version']}")
        
        # Check for OpenAI modules
        openai_modules = []
        if 'modules' in meta:
            modules = meta['modules']
            openai_modules = [m for m in modules.keys() if 'openai' in m]
            
            if openai_modules:
                logger.info(f"Found {len(openai_modules)} OpenAI modules:")
                for module in sorted(openai_modules):
                    logger.info(f"  - {module}")
            else:
                logger.warning("No OpenAI modules found")
        
        return openai_modules
    except Exception as e:
        logger.error(f"Error checking OpenAI modules: {e}")
        return []


def setup_collection(client):
    """Set up a collection with text2vec-openai vectorizer."""
    try:
        # Try to get the collection to see if it exists
        logger.info(f"Checking if collection {COLLECTION_NAME} exists")
        
        try:
            # Use the REST API directly
            response = client._connection.get(f"v1/collections/{COLLECTION_NAME}")
            
            if response.status_code == 200:
                # Collection exists, delete it
                logger.info(f"Collection {COLLECTION_NAME} exists, deleting it")
                delete_response = client._connection.delete(f"v1/collections/{COLLECTION_NAME}")
                
                if delete_response.status_code == 200:
                    logger.info(f"Collection {COLLECTION_NAME} deleted successfully")
                else:
                    logger.warning(f"Failed to delete collection: {delete_response.status_code} - {delete_response.text}")
                    time.sleep(1)  # Wait a bit before trying to create
            elif response.status_code != 404:
                logger.warning(f"Unexpected response when checking collection: {response.status_code} - {response.text}")
        except Exception as e:
            logger.info(f"Collection check failed, assuming it doesn't exist: {e}")
        
        # Create a new collection with text2vec-openai vectorizer
        logger.info(f"Creating collection {COLLECTION_NAME} with text2vec-openai vectorizer")
        
        collection_obj = {
            "name": COLLECTION_NAME,
            "description": "AI-related articles for demonstration",
            "vectorizer": "text2vec-openai",
            "vectorizerConfig": {
                "model": "text-embedding-3-small",
                "type": "text"
            },
            "moduleConfig": {
                "generative-openai": {
                    "model": "gpt-3.5-turbo"
                }
            },
            "properties": [
                {
                    "name": "title",
                    "dataType": ["text"],
                    "description": "Title of the article",
                    "moduleConfig": {
                        "text2vec-openai": {
                            "skip": False,
                            "vectorizePropertyName": False
                        }
                    }
                },
                {
                    "name": "content",
                    "dataType": ["text"],
                    "description": "Content of the article",
                    "moduleConfig": {
                        "text2vec-openai": {
                            "skip": False,
                            "vectorizePropertyName": False
                        }
                    }
                },
                {
                    "name": "category",
                    "dataType": ["text"],
                    "description": "Category of the article",
                    "moduleConfig": {
                        "text2vec-openai": {
                            "skip": True
                        }
                    }
                }
            ]
        }
        
        response = client._connection.post("v1/collections", collection_obj)
        
        if response.status_code == 200:
            logger.info(f"Collection {COLLECTION_NAME} created successfully")
            return True
        else:
            logger.error(f"Failed to create collection: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        logger.error(f"Error setting up collection: {e}")
        return False


def add_sample_data(client):
    """Add sample data to the collection."""
    try:
        sample_data = [
            {
                "title": "Understanding Large Language Models",
                "content": "Large Language Models (LLMs) are deep learning algorithms that can recognize, summarize, translate, predict, and generate text and other content based on knowledge gained from massive datasets.",
                "category": "AI Technology"
            },
            {
                "title": "Vector Databases for AI Applications",
                "content": "Vector databases are specialized database systems designed to store and query high-dimensional vectors efficiently. They are crucial for AI applications that rely on embeddings.",
                "category": "AI Infrastructure"
            },
            {
                "title": "The Ethics of Artificial Intelligence",
                "content": "AI ethics involves designing and using AI systems responsibly and ethically. Key concerns include bias and fairness, privacy, transparency, and accountability.",
                "category": "AI Ethics"
            }
        ]
        
        for idx, data in enumerate(sample_data):
            logger.info(f"Adding article {idx+1}: {data['title']}")
            
            # Generate a UUID for the object
            object_uuid = str(uuid.uuid4())
            
            # Add the data object
            response = client._connection.post(
                f"v1/collections/{COLLECTION_NAME}/objects",
                {
                    "id": object_uuid,
                    "properties": data
                }
            )
            
            if response.status_code == 200:
                logger.info(f"Article {idx+1} added successfully")
            else:
                logger.warning(f"Failed to add article {idx+1}: {response.status_code} - {response.text}")
        
        # Wait for indexing to complete
        logger.info("Waiting for indexing to complete...")
        time.sleep(2)
        
        return True
    except Exception as e:
        logger.error(f"Error adding sample data: {e}")
        return False


def demonstrate_vector_search(client):
    """Demonstrate vector search with text2vec-openai."""
    logger.info("\n=== DEMONSTRATING VECTOR SEARCH WITH TEXT2VEC-OPENAI ===\n")
    
    try:
        query = "What are the ethical considerations in AI?"
        logger.info(f"Performing semantic search with query: '{query}'")
        
        response = client._connection.post(
            "v1/graphql",
            {
                "query": f"""
                {{
                  Get {{
                    {COLLECTION_NAME}(
                      nearText: {{
                        concepts: ["{query}"]
                        certainty: 0.7
                      }}
                      limit: 2
                    ) {{
                      title
                      content
                      category
                      _additional {{
                        certainty
                      }}
                    }}
                  }}
                }}
                """
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            
            if "data" in data and "Get" in data["data"] and COLLECTION_NAME in data["data"]["Get"]:
                articles = data["data"]["Get"][COLLECTION_NAME]
                logger.info(f"Found {len(articles)} relevant articles:")
                
                for idx, article in enumerate(articles):
                    certainty = article["_additional"]["certainty"] if "_additional" in article and "certainty" in article["_additional"] else "N/A"
                    logger.info(f"  {idx+1}. {article['title']} (Category: {article['category']}, Certainty: {certainty})")
                    logger.info(f"     Content: {article['content'][:100]}...")
            else:
                logger.warning("No results found or unexpected response structure")
        else:
            logger.warning(f"Search failed: {response.status_code} - {response.text}")
    except Exception as e:
        logger.error(f"Error demonstrating vector search: {e}")


def demonstrate_generative_search(client):
    """Demonstrate generative search with generative-openai."""
    logger.info("\n=== DEMONSTRATING GENERATIVE AI WITH GENERATIVE-OPENAI ===\n")
    
    try:
        query = "What are vector databases?"
        logger.info(f"Performing generative search with query: '{query}'")
        
        response = client._connection.post(
            "v1/graphql",
            {
                "query": f"""
                {{
                  Get {{
                    {COLLECTION_NAME}(
                      nearText: {{
                        concepts: ["{query}"]
                      }}
                      limit: 1
                    ) {{
                      title
                      content
                      _additional {{
                        generate(
                          singleResult: {{
                            prompt: "Explain this in simpler terms:"
                          }}
                        ) {{
                          singleResult
                          error
                        }}
                      }}
                    }}
                  }}
                }}
                """
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            
            if "data" in data and "Get" in data["data"] and COLLECTION_NAME in data["data"]["Get"]:
                articles = data["data"]["Get"][COLLECTION_NAME]
                
                if articles:
                    article = articles[0]
                    logger.info(f"Article: {article['title']}")
                    logger.info(f"Original content: {article['content']}")
                    
                    if "_additional" in article and "generate" in article["_additional"]:
                        generated = article["_additional"]["generate"]
                        
                        if "singleResult" in generated:
                            logger.info(f"\nSimplified explanation: {generated['singleResult']}")
                        elif "error" in generated:
                            logger.error(f"Generation error: {generated['error']}")
                    else:
                        logger.warning("No generated content found")
                else:
                    logger.warning("No articles found")
            else:
                logger.warning("No results found or unexpected response structure")
        else:
            logger.warning(f"Generate query failed: {response.status_code} - {response.text}")
    except Exception as e:
        logger.error(f"Error demonstrating generative search: {e}")


def demonstrate_standalone_generation(client):
    """Demonstrate standalone text generation with generative-openai."""
    logger.info("\n=== DEMONSTRATING STANDALONE GENERATIVE AI ===\n")
    
    try:
        prompt = "Explain the concept of vector embeddings in AI in 3 sentences."
        logger.info(f"Using generative-openai module with prompt: '{prompt}'")
        
        response = client._connection.post(
            "v1/modules/generative-openai/generate",
            {
                "prompt": prompt
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            
            if "result" in data:
                logger.info(f"Generated response: {data['result']}")
            else:
                logger.warning("No result found in response")
        else:
            logger.warning(f"Standalone generate failed: {response.status_code} - {response.text}")
    except Exception as e:
        logger.error(f"Error demonstrating standalone generation: {e}")


def main():
    """Main function to demonstrate OpenAI modules in Weaviate."""
    logger.info("Starting OpenAI modules demonstration")
    
    client = None
    
    try:
        # Connect to Weaviate
        client = connect_to_weaviate()
        
        if not client:
            logger.error("Failed to connect to Weaviate")
            sys.exit(1)
        
        # Check for OpenAI modules
        openai_modules = check_openai_modules(client)
        
        if not openai_modules or "text2vec-openai" not in openai_modules or "generative-openai" not in openai_modules:
            logger.error("Required OpenAI modules not available")
            sys.exit(1)
        
        # Set up collection with text2vec-openai vectorizer
        if not setup_collection(client):
            logger.error("Failed to set up collection")
            sys.exit(1)
        
        # Add sample data
        if not add_sample_data(client):
            logger.error("Failed to add sample data")
            sys.exit(1)
        
        # Demonstrate vector search with text2vec-openai
        demonstrate_vector_search(client)
        
        # Demonstrate generative search with generative-openai
        demonstrate_generative_search(client)
        
        # Demonstrate standalone text generation with generative-openai
        demonstrate_standalone_generation(client)
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
    finally:
        # Clean up
        if client:
            client.close()
        logger.info("OpenAI modules demonstration completed")


if __name__ == "__main__":
    main()
