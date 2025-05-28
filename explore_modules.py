#!/usr/bin/env python3

import os
import sys
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
logger = logging.getLogger("WeaviateModulesExplorer")

# Get the project root directory
ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.append(ROOT)

# Load environment variables from .env file
load_dotenv(os.path.join(ROOT, '.env'))
weaviate_url = os.getenv("WEAVIATE_URL")
weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

if not weaviate_url or not weaviate_api_key:
    logger.error("Missing required environment variables: WEAVIATE_URL or WEAVIATE_API_KEY")
    sys.exit(1)


def main():
    """Main function to explore Weaviate modules."""
    logger.info("Starting Weaviate modules exploration")
    
    # Ensure the URL has https:// prefix
    global weaviate_url
    if not weaviate_url.startswith('https://'):
        weaviate_url = f"https://{weaviate_url}"
    
    logger.info(f"Connecting to Weaviate at {weaviate_url}")
    
    try:
        # Connect to Weaviate
        headers = {}
        if openai_api_key:
            headers["X-OpenAI-Api-Key"] = openai_api_key
        
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=weaviate_url,
            auth_credentials=AuthApiKey(api_key=weaviate_api_key),
            headers=headers
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
        
        # Extract and print modules
        if 'modules' in meta:
            modules = meta['modules']
            logger.info(f"Found {len(modules)} modules")
            
            # Group modules by type
            module_groups = {}
            
            for module_name in modules:
                prefix = module_name.split('-')[0] if '-' in module_name else "other"
                if prefix not in module_groups:
                    module_groups[prefix] = []
                module_groups[prefix].append(module_name)
            
            # Print modules by group
            for group, group_modules in sorted(module_groups.items()):
                logger.info(f"\n=== {group.upper()} MODULES ===")
                for module in sorted(group_modules):
                    logger.info(f"  - {module}")
                    
                    # Print module details if available
                    if modules[module]:
                        logger.info(f"    Details: {json.dumps(modules[module], indent=2)}")
        else:
            logger.warning("No modules found in Weaviate meta information")
        
        # Try to get schema information
        logger.info("\n=== TRYING TO EXPLORE SCHEMA ===")
        
        # Try different API paths for schema
        api_paths = [
            "v1/schema",
            "schema",
            "v1/collections",
            "collections"
        ]
        
        for path in api_paths:
            try:
                logger.info(f"Trying to get schema via {path}")
                response = client._connection.get(path)
                
                if response.status_code == 200:
                    logger.info(f"Successfully retrieved schema via {path}")
                    schema_data = response.json()
                    logger.info(f"Schema data: {json.dumps(schema_data, indent=2)[:500]}...")  # Truncate for readability
                    break
                else:
                    logger.warning(f"Failed to get schema via {path}: {response.status_code} - {response.text}")
            except Exception as e:
                logger.error(f"Error getting schema via {path}: {e}")
        
        # Try to use GraphQL API
        logger.info("\n=== TRYING GRAPHQL META QUERY ===")
        
        # Try different API paths for GraphQL
        graphql_paths = [
            "v1/graphql",
            "graphql"
        ]
        
        for path in graphql_paths:
            try:
                logger.info(f"Trying GraphQL query via {path}")
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
                    logger.info(f"Successfully executed GraphQL query via {path}")
                    graphql_data = response.json()
                    logger.info(f"GraphQL data: {json.dumps(graphql_data, indent=2)}")
                    break
                else:
                    logger.warning(f"Failed to execute GraphQL query via {path}: {response.status_code} - {response.text}")
            except Exception as e:
                logger.error(f"Error executing GraphQL query via {path}: {e}")
        
        # Print summary of OpenAI modules
        logger.info("\n=== OPENAI MODULES SUMMARY ===")
        
        openai_modules = [m for m in modules.keys() if 'openai' in m]
        
        if openai_modules:
            logger.info(f"Found {len(openai_modules)} OpenAI modules:")
            for module in sorted(openai_modules):
                logger.info(f"  - {module}")
                
                # Print module details if available
                if modules[module]:
                    logger.info(f"    Details: {json.dumps(modules[module], indent=2)}")
        else:
            logger.warning("No OpenAI modules found")
        
        # Print usage recommendations
        logger.info("\n=== USAGE RECOMMENDATIONS ===")
        
        logger.info("Based on the available modules, you can use the following in your applications:")
        
        if 'text2vec-openai' in modules:
            logger.info("1. text2vec-openai:")
            logger.info("   - Use for text vectorization when creating collections")
            logger.info("   - Provides semantic search capabilities")
            logger.info("   - Example usage in collection creation:")
            logger.info("""
            collection_obj = {
                "name": "MyCollection",
                "vectorizer": "text2vec-openai",
                "vectorizerConfig": {
                    "model": "text-embedding-3-small",
                    "type": "text"
                }
            }
            """)
        
        if 'generative-openai' in modules:
            logger.info("2. generative-openai:")
            logger.info("   - Use for text generation in GraphQL queries")
            logger.info("   - Provides summarization, explanation, and other generative capabilities")
            logger.info("   - Example usage in GraphQL query:")
            logger.info("""
            {
              Get {
                MyCollection(
                  nearText: {
                    concepts: ["my search query"]
                  }
                  limit: 1
                ) {
                  title
                  content
                  _additional {
                    generate(
                      singleResult: {
                        prompt: "Explain this in simpler terms:"
                      }
                    ) {
                      singleResult
                    }
                  }
                }
              }
            }
            """)
        
        if 'qna-openai' in modules:
            logger.info("3. qna-openai:")
            logger.info("   - Use for question answering over your data")
            logger.info("   - Provides direct answers to questions based on your data")
            logger.info("   - Example usage in GraphQL query:")
            logger.info("""
            {
              Get {
                MyCollection(
                  nearText: {
                    concepts: ["my search context"]
                  }
                  limit: 5
                ) {
                  title
                  content
                  _additional {
                    answer(
                      question: "What is the main point?"
                    ) {
                      result
                    }
                  }
                }
              }
            }
            """)
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
    finally:
        # Clean up
        if 'client' in locals() and client:
            client.close()
        logger.info("Weaviate modules exploration completed")


if __name__ == "__main__":
    main()
