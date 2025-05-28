#!/usr/bin/env python3

import os
import sys
import time
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
logger = logging.getLogger("WeaviateClientDemo")

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

# Define the class name for our demo
CLASS_NAME = "AIArticles"

def main():
    """Main function to demonstrate Weaviate client API."""
    logger.info("Starting Weaviate client API demonstration")
    
    client = None
    
    try:
        # Connect to Weaviate
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
        
        # Try to get schema information
        try:
            logger.info("Attempting to get schema information")
            schema = client.get_schema()
            
            if schema:
                logger.info(f"Schema: {json.dumps(schema, indent=2)}")
            else:
                logger.info("No schema found")
        except Exception as e:
            logger.error(f"Error getting schema: {e}")
            logger.info("Continuing with the demo...")
        
        # Try to use the client's query builder
        logger.info("\n=== DEMONSTRATING STANDALONE GENERATIVE AI ===\n")
        
        prompt = "Explain the concept of vector embeddings in AI in 3 sentences."
        logger.info(f"Using generative-openai module with prompt: '{prompt}'")
        
        try:
            # Try using the client's generate method if available
            try:
                result = client.generate.generate_text(
                    prompt=prompt,
                    model="gpt-3.5-turbo"
                )
                logger.info(f"Generated response: {result}")
            except AttributeError:
                # If the generate method is not available, try using the modules endpoint directly
                logger.info("Client generate method not available, trying direct API call")
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
                    logger.warning(f"Generate failed: {response.status_code} - {response.text}")
                    
                    # Try alternative path
                    logger.info("Trying alternative API path")
                    response = client._connection.post(
                        "modules/generative-openai/generate",
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
                        logger.warning(f"Generate failed: {response.status_code} - {response.text}")
        except Exception as e:
            logger.error(f"Error using generative AI: {e}")
        
        # Try to use the client's query builder for GraphQL
        logger.info("\n=== TRYING DIFFERENT GRAPHQL QUERY APPROACHES ===\n")
        
        try:
            # Try using the client's query builder if available
            logger.info("Attempting to use client query builder")
            
            try:
                # Method 1: Using the client's query builder
                result = (
                    client.query
                    .get("Article", ["title", "content"])
                    .with_limit(2)
                    .do()
                )
                logger.info(f"Query result: {json.dumps(result, indent=2)}")
            except AttributeError:
                logger.info("Client query builder not available or no data found")
                
                # Method 2: Direct GraphQL query
                logger.info("Trying direct GraphQL query")
                
                # Try different paths for GraphQL endpoint
                paths = ["v1/graphql", "graphql"]
                
                for path in paths:
                    try:
                        response = client._connection.post(
                            path,
                            {
                                "query": """
                                {
                                  Meta {
                                    modules {
                                      module
                                      version
                                    }
                                  }
                                }
                                """
                            }
                        )
                        
                        if response.status_code == 200:
                            data = response.json()
                            logger.info(f"GraphQL Meta query result via {path}: {json.dumps(data, indent=2)}")
                            break
                        else:
                            logger.warning(f"GraphQL query via {path} failed: {response.status_code} - {response.text}")
                    except Exception as e:
                        logger.error(f"Error with GraphQL query via {path}: {e}")
        except Exception as e:
            logger.error(f"Error using GraphQL: {e}")
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
    finally:
        # Clean up
        if client:
            client.close()
        logger.info("Weaviate client API demonstration completed")


if __name__ == "__main__":
    main()
