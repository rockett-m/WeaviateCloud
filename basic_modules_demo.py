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
logger = logging.getLogger("BasicModulesDemo")

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


def connect_to_weaviate():
    """Connect to Weaviate Cloud instance."""
    logger.info(f"Connecting to Weaviate at {weaviate_url}")
    
    headers = {}
    if openai_api_key:
        headers["X-OpenAI-Api-Key"] = openai_api_key
    
    try:
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
            return None
        
        return client
    except Exception as e:
        logger.error(f"Error connecting to Weaviate: {e}")
        return None


def get_available_modules(client):
    """Get available modules from Weaviate."""
    try:
        meta = client.get_meta()
        
        # Print Weaviate version
        if 'version' in meta:
            logger.info(f"Weaviate version: {meta['version']}")
        
        # Extract and print modules
        modules = {}
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
                logger.info(f"  {group.upper()} MODULES:")
                for module in sorted(group_modules):
                    logger.info(f"    - {module}")
        else:
            logger.warning("No modules found in Weaviate meta information")
        
        return modules
    except Exception as e:
        logger.error(f"Error getting modules: {e}")
        return {}


def explore_openai_modules(modules):
    """Explore OpenAI modules specifically."""
    logger.info("\n=== EXPLORING OPENAI MODULES ===\n")
    
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


def main():
    """Main function to demonstrate basic Weaviate module exploration."""
    logger.info("Starting basic Weaviate modules demonstration")
    
    client = None
    
    try:
        # Connect to Weaviate
        client = connect_to_weaviate()
        
        if not client:
            logger.error("Failed to connect to Weaviate")
            sys.exit(1)
        
        # Get available modules
        modules = get_available_modules(client)
        
        # Explore OpenAI modules specifically
        explore_openai_modules(modules)
        
        # Print a summary of the available modules
        logger.info("\n=== MODULE AVAILABILITY SUMMARY ===\n")
        
        # Check for specific modules of interest
        modules_of_interest = [
            "text2vec-openai",      # For text vectorization
            "generative-openai",     # For text generation
            "qna-openai",            # For question answering
            "text2vec-huggingface",  # Alternative vectorizer
            "generative-cohere",     # Alternative generator
            "multi2vec-clip"         # For multi-modal (text+image)
        ]
        
        for module in modules_of_interest:
            status = "✅ Available" if module in modules else "❌ Not available"
            logger.info(f"{module}: {status}")
        
        logger.info("\nTo use these modules in your applications:")
        logger.info("1. For text2vec-openai: Set it as the vectorizer when creating a class/collection")
        logger.info("2. For generative-openai: Use it in GraphQL queries with the generate() function")
        logger.info("3. For qna-openai: Use it for question answering over your data")
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
    finally:
        # Clean up
        if client:
            client.close()
        logger.info("Basic Weaviate modules demonstration completed")


if __name__ == "__main__":
    main()
