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
logger = logging.getLogger("WeaviateCollectionsDemo")

# Get the project root directory
ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.append(ROOT)

# Load environment variables
load_dotenv()
weaviate_url = os.getenv("WEAVIATE_URL")
weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

if not weaviate_url or not weaviate_api_key or not openai_api_key:
    logger.error("Missing required environment variables: WEAVIATE_URL, WEAVIATE_API_KEY, or OPENAI_API_KEY")
    sys.exit(1)

# Define the collection name for our demo
COLLECTION_NAME = "AIArticles"

def main():
    """Main function to demonstrate Weaviate collections API."""
    logger.info("Starting Weaviate collections API demonstration")
    
    client = None
    
    try:
        # Connect to Weaviate
        # Ensure the URL has https:// prefix
        if not weaviate_url.startswith('https://'):
            weaviate_url = f"https://{weaviate_url}"
            
        logger.info(f"Connecting to Weaviate at {weaviate_url}")
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
            sys.exit(1)
        
        # Get meta information
        logger.info("Retrieving meta information")
        meta = client.get_meta()
        
        # Print Weaviate version
        if 'version' in meta:
            logger.info(f"Weaviate version: {meta['version']}")
        
        # Check available modules
        if 'modules' in meta:
            modules = meta['modules']
            logger.info(f"Found {len(modules)} modules")
            
            # Check if the required OpenAI modules are available
            if 'text2vec-openai' in modules:
                logger.info("text2vec-openai module is available")
            else:
                logger.warning("text2vec-openai module is not available")
                
            if 'generative-openai' in modules:
                logger.info("generative-openai module is available")
            else:
                logger.warning("generative-openai module is not available")
        
        # List collections using the collections API
        logger.info("Listing collections")
        response = client._connection.get("collections")
        
        if response.status_code == 200:
            collections_data = response.json()
            
            if 'collections' in collections_data:
                collections = collections_data['collections']
                logger.info(f"Found {len(collections)} collections:")
                
                for collection in collections:
                    if 'name' in collection:
                        logger.info(f"  - {collection['name']}")
            else:
                logger.info("No collections found")
        else:
            logger.warning(f"Failed to list collections: {response.status_code} - {response.text}")
        
        # Check if our collection exists and delete it if it does
        logger.info(f"Checking if collection {COLLECTION_NAME} exists")
        response = client._connection.get(f"collections/{COLLECTION_NAME}")
        
        if response.status_code == 200:
            logger.info(f"Collection {COLLECTION_NAME} exists, deleting it")
            delete_response = client._connection.delete(f"collections/{COLLECTION_NAME}")
            
            if delete_response.status_code == 200:
                logger.info(f"Collection {COLLECTION_NAME} deleted successfully")
            else:
                logger.warning(f"Failed to delete collection: {delete_response.status_code} - {delete_response.text}")
                # Wait a bit before trying to create the collection
                time.sleep(1)
        elif response.status_code == 404:
            logger.info(f"Collection {COLLECTION_NAME} does not exist")
        else:
            logger.warning(f"Unexpected response when checking collection: {response.status_code} - {response.text}")
        
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
        
        response = client._connection.post("collections", collection_obj)
        
        if response.status_code == 200:
            logger.info(f"Collection {COLLECTION_NAME} created successfully")
        else:
            logger.error(f"Failed to create collection: {response.status_code} - {response.text}")
            sys.exit(1)
        
        # Add sample data
        logger.info("Adding sample data")
        
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
                f"collections/{COLLECTION_NAME}/objects",
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
        
        # Perform a vector search
        logger.info("\n=== DEMONSTRATING VECTOR SEARCH WITH TEXT2VEC-OPENAI ===\n")
        
        query = "What are the ethical considerations in AI?"
        logger.info(f"Performing semantic search with query: '{query}'")
        
        search_response = client._connection.post(
            "graphql",
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
        
        if search_response.status_code == 200:
            search_data = search_response.json()
            
            if "data" in search_data and "Get" in search_data["data"] and COLLECTION_NAME in search_data["data"]["Get"]:
                articles = search_data["data"]["Get"][COLLECTION_NAME]
                logger.info(f"Found {len(articles)} relevant articles:")
                
                for idx, article in enumerate(articles):
                    certainty = article["_additional"]["certainty"] if "_additional" in article and "certainty" in article["_additional"] else "N/A"
                    logger.info(f"  {idx+1}. {article['title']} (Category: {article['category']}, Certainty: {certainty})")
                    logger.info(f"     Content: {article['content'][:100]}...")
            else:
                logger.warning("No results found or unexpected response structure")
        else:
            logger.warning(f"Search failed: {search_response.status_code} - {search_response.text}")
        
        # Perform a generative query
        logger.info("\n=== DEMONSTRATING GENERATIVE AI WITH GENERATIVE-OPENAI ===\n")
        
        query = "What are vector databases?"
        logger.info(f"Performing generative search with query: '{query}'")
        
        generate_response = client._connection.post(
            "graphql",
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
        
        if generate_response.status_code == 200:
            generate_data = generate_response.json()
            
            if "data" in generate_data and "Get" in generate_data["data"] and COLLECTION_NAME in generate_data["data"]["Get"]:
                articles = generate_data["data"]["Get"][COLLECTION_NAME]
                
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
            logger.warning(f"Generate query failed: {generate_response.status_code} - {generate_response.text}")
        
        # Use the standalone generative API
        logger.info("\n=== DEMONSTRATING STANDALONE GENERATIVE AI ===\n")
        
        prompt = "Explain the concept of vector embeddings in AI in 3 sentences."
        logger.info(f"Using generative-openai module with prompt: '{prompt}'")
        
        standalone_response = client._connection.post(
            "modules/generative-openai/generate",
            {
                "prompt": prompt
            }
        )
        
        if standalone_response.status_code == 200:
            standalone_data = standalone_response.json()
            
            if "result" in standalone_data:
                logger.info(f"Generated response: {standalone_data['result']}")
            else:
                logger.warning("No result found in response")
        else:
            logger.warning(f"Standalone generate failed: {standalone_response.status_code} - {standalone_response.text}")
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
    finally:
        # Clean up
        if client:
            client.close()
        logger.info("Weaviate collections API demonstration completed")


if __name__ == "__main__":
    main()
