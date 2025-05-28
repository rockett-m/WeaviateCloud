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
logger = logging.getLogger("WeaviateQueryDemo")

# Get the project root directory
ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.append(ROOT)

# Load environment variables
load_dotenv()
weaviate_url = os.getenv("WEAVIATE_URL")
weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

if not weaviate_url or not weaviate_api_key:
    logger.error("Missing required environment variables: WEAVIATE_URL or WEAVIATE_API_KEY")
    sys.exit(1)


def main():
    """Main function to demonstrate querying Weaviate."""
    logger.info("Starting Weaviate query demonstration")
    
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
        
        # Extract and print modules
        if 'modules' in meta:
            modules = meta['modules']
            logger.info(f"Found {len(modules)} modules:")
            
            # Group modules by type
            module_groups = {}
            
            for module_name in modules:
                prefix = module_name.split('-')[0] if '-' in module_name else "other"
                if prefix not in module_groups:
                    module_groups[prefix] = []
                module_groups[prefix].append(module_name)
            
            # Print modules by group
            for group, group_modules in module_groups.items():
                logger.info(f"  {group.upper()} MODULES:")
                for module in sorted(group_modules):
                    logger.info(f"    - {module}")
        else:
            logger.warning("No modules found in Weaviate meta information")
        
        # Try to get schema information using direct REST API
        try:
            logger.info("Attempting to get schema information")
            response = client._connection.get("schema")
            
            if response.status_code == 200:
                schema = response.json()
                
                if 'classes' in schema and schema['classes']:
                    classes = schema['classes']
                    logger.info(f"Found {len(classes)} classes:")
                    for cls in classes:
                        logger.info(f"  - {cls['class']}: {cls.get('description', 'No description')}")
                        
                        # Print vectorizer if available
                        if 'vectorizer' in cls:
                            logger.info(f"    Vectorizer: {cls['vectorizer']}")
                        
                        # Print properties if available
                        if 'properties' in cls:
                            logger.info(f"    Properties:")
                            for prop in cls['properties']:
                                logger.info(f"      - {prop['name']}: {prop['dataType']}")
                else:
                    logger.info("No classes found in schema")
            else:
                logger.warning(f"Failed to get schema: {response.status_code} - {response.text}")
                
        except Exception as e:
            logger.error(f"Error getting schema via REST API: {e}")
        
        # Try to perform a simple query if there are classes
        try:
            if 'classes' in schema and schema['classes']:
                # Get the first class to query
                first_class = schema['classes'][0]['class']
                logger.info(f"\nAttempting to query class: {first_class}")
                
                # Use the query API
                result = client._connection.post(
                    "graphql",
                    {
                        "query": f"""
                        {{
                          Get {{
                            {first_class}(limit: 3) {{
                              _additional {{
                                id
                              }}
                            }}
                          }}
                        }}
                        """
                    }
                )
                
                if result.status_code == 200:
                    data = result.json()
                    
                    if "data" in data and "Get" in data["data"] and first_class in data["data"]["Get"]:
                        objects = data["data"]["Get"][first_class]
                        logger.info(f"Found {len(objects)} objects in class {first_class}")
                        
                        # Print object IDs
                        for idx, obj in enumerate(objects):
                            if "_additional" in obj and "id" in obj["_additional"]:
                                logger.info(f"  Object {idx+1} ID: {obj['_additional']['id']}")
                    else:
                        logger.warning(f"No objects found in class {first_class}")
                else:
                    logger.warning(f"Query failed: {result.status_code} - {result.text}")
        except Exception as e:
            logger.error(f"Error querying data: {e}")
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
    finally:
        # Clean up
        if client:
            client.close()
        logger.info("Weaviate query demonstration completed")


if __name__ == "__main__":
    main()
