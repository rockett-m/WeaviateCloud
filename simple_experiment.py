#!/usr/bin/env python3

import os
import sys
import time
import logging
import json
import uuid
from typing import Dict, List, Any
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
logger = logging.getLogger("WeaviateExperiment")

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


def connect_to_weaviate():
    """Connect to Weaviate Cloud and return the client."""
    try:
        logger.info(f"Connecting to Weaviate at {weaviate_url}")
        
        # Connect to Weaviate using the client API
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
            
        return client
            
    except Exception as e:
        logger.error(f"Error connecting to Weaviate: {e}")
        sys.exit(1)


def get_available_modules(client):
    """Get all available modules from the Weaviate instance."""
    try:
        logger.info("Retrieving available modules from Weaviate")
        
        # Get meta information
        meta = client.get_meta()
        
        # Extract modules from meta response
        if 'modules' in meta:
            modules = list(meta['modules'].keys())
            logger.info(f"Found {len(modules)} modules: {', '.join(modules)}")
            return modules
        else:
            logger.warning("No modules found in Weaviate meta information")
            return []
            
    except Exception as e:
        logger.error(f"Error retrieving modules: {e}")
        return []


def list_schema(client):
    """List the current schema in Weaviate."""
    try:
        logger.info("Retrieving current schema from Weaviate")
        
        # Get schema
        schema = client.schema.get()
        
        # Print classes
        if 'classes' in schema and schema['classes']:
            classes = schema['classes']
            logger.info(f"Found {len(classes)} classes:")
            for cls in classes:
                logger.info(f"  - {cls['class']}: {cls.get('description', 'No description')}")
                if 'properties' in cls:
                    for prop in cls['properties']:
                        logger.info(f"    - {prop['name']}: {prop['dataType']}")
        else:
            logger.info("No classes found in schema")
            
    except Exception as e:
        logger.error(f"Error retrieving schema: {e}")


def experiment_with_openai_modules(client):
    """Experiment with OpenAI modules (text2vec-openai and generative-openai)."""
    class_name = "AIExperiment"
    
    try:
        # Check if class already exists and delete it
        schema = client.schema.get()
        classes = [cls['class'] for cls in schema['classes']] if 'classes' in schema else []
        
        if class_name in classes:
            logger.info(f"Deleting existing class: {class_name}")
            client.schema.delete_class(class_name)
        
        # Create class with text2vec-openai vectorizer
        logger.info("Creating schema with text2vec-openai vectorizer")
        
        class_obj = {
            "class": class_name,
            "description": "Test class for OpenAI modules",
            "vectorizer": "text2vec-openai",
            "moduleConfig": {
                "text2vec-openai": {
                    "model": "ada",
                    "modelVersion": "002",
                    "type": "text"
                },
                "generative-openai": {
                    "model": "gpt-3.5-turbo"
                }
            },
            "properties": [
                {
                    "name": "title",
                    "dataType": ["text"],
                    "description": "Title of the document",
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
                    "description": "Content of the document",
                    "moduleConfig": {
                        "text2vec-openai": {
                            "skip": False,
                            "vectorizePropertyName": False
                        }
                    }
                }
            ]
        }
        
        # Create the class
        client.schema.create_class(class_obj)
        logger.info(f"Schema created successfully: {class_name}")
        
        # Add test data
        test_data = [
            {
                "title": "Artificial Intelligence",
                "content": "Artificial intelligence is the simulation of human intelligence processes by machines."
            },
            {
                "title": "Machine Learning",
                "content": "Machine learning is a method of data analysis that automates analytical model building."
            },
            {
                "title": "Vector Databases",
                "content": "Vector databases store data as high-dimensional vectors, enabling efficient similarity search."
            }
        ]
        
        logger.info(f"Adding {len(test_data)} test objects to {class_name}")
        
        # Use batch processing to add data
        with client.batch as batch:
            batch.batch_size = 10
            
            for idx, data in enumerate(test_data):
                logger.info(f"Adding object {idx+1}: {data['title']}")
                
                # Generate a UUID for the object
                object_uuid = str(uuid.uuid4())
                
                # Add the data object to the batch
                batch.add_data_object(
                    data_object=data,
                    class_name=class_name,
                    uuid=object_uuid
                )
        
        logger.info("Test data added successfully")
        
        # Wait for indexing to complete
        time.sleep(2)
        
        # Perform a vector search
        query = "What is AI?"
        logger.info(f"Performing vector search with query: '{query}'")
        
        result = client.query.get(
            class_name, ["title", "content"]
        ).with_near_text({
            "concepts": [query]
        }).with_limit(2).do()
        
        # Process and log results
        if "data" in result and "Get" in result["data"]:
            objects = result["data"]["Get"][class_name]
            logger.info(f"Found {len(objects)} results")
            
            for idx, obj in enumerate(objects):
                logger.info(f"Result {idx+1}: {obj['title']} - {obj['content'][:50]}...")
        else:
            logger.warning(f"No results found or unexpected response structure")
            logger.debug(f"Response: {json.dumps(result, indent=2)}")
        
        # Perform a generative query
        logger.info("Performing generative query with generative-openai")
        
        result = client.query.get(
            class_name, ["title", "content"]
        ).with_generate(
            single_prompt="Summarize this text:",
            properties=["content"]
        ).with_limit(2).do()
        
        # Process and log results
        if "data" in result and "Get" in result["data"]:
            objects = result["data"]["Get"][class_name]
            logger.info(f"Found {len(objects)} results")
            
            for idx, obj in enumerate(objects):
                if "_additional" in obj and "generate" in obj["_additional"]:
                    generated_text = obj["_additional"]["generate"]["singleResult"]
                    logger.info(f"Generated summary for '{obj['title']}': {generated_text}")
                else:
                    logger.warning(f"No generated text for {obj['title']}")
        else:
            logger.warning(f"No results found or unexpected response structure")
            logger.debug(f"Response: {json.dumps(result, indent=2)}")
        
    except Exception as e:
        logger.error(f"Error in OpenAI modules experiment: {e}")


def main():
    """Main function to run the Weaviate module experiments."""
    logger.info("Starting Weaviate module experiments")
    
    client = None
    
    try:
        # Connect to Weaviate
        client = connect_to_weaviate()
        
        # Get available modules
        modules = get_available_modules(client)
        
        # List current schema
        list_schema(client)
        
        # Experiment with OpenAI modules if available
        if "text2vec-openai" in modules and "generative-openai" in modules:
            logger.info("\n==== Testing OpenAI Modules ====")
            experiment_with_openai_modules(client)
        else:
            logger.warning("OpenAI modules not available, skipping experiment")
        
    except KeyboardInterrupt:
        logger.info("Experiment interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
    finally:
        # Clean up
        if client:
            client.close()
        logger.info("Weaviate module experiments completed")


if __name__ == "__main__":
    main()
