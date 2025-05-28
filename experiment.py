#!/usr/bin/env python3

import os
import sys
import logging
import json
import time
import uuid
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
import weaviate
from weaviate.auth import AuthApiKey
from weaviate.exceptions import WeaviateQueryError, WeaviateConnectionError

# Import the available modules - these need to be imported as weaviate.module
# Based on the screenshot, these are the modules available in Weaviate Cloud

# Get the project root directory
ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.append(ROOT)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(ROOT, "experiment.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("WeaviateExperiment")

# Load environment variables
load_dotenv()
weaviate_url = os.getenv("WEAVIATE_URL")
weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

if not weaviate_url or not weaviate_api_key:
    logger.error("Missing required environment variables: WEAVIATE_URL or WEAVIATE_API_KEY")
    sys.exit(1)

# Define available modules based on the screenshot
GENERATIVE_MODULES = [
    "generative-anthropic", "generative-anyscale", "generative-aws", "generative-cohere",
    "generative-databricks", "generative-friendliai", "generative-mistral", "generative-nvidia",
    "generative-octoai", "generative-ollama", "generative-openai", "generative-palm", "generative-xai"
]

MULTI2VEC_MODULES = [
    "multi2vec-cohere", "multi2vec-jinaai", "multi2vec-nvidia", "multi2vec-palm", "multi2vec-voyageai"
]

QNA_MODULES = ["qna-openai"]

REF2VEC_MODULES = ["ref2vec-centroid"]

RERANKER_MODULES = [
    "reranker-cohere", "reranker-jinaai", "reranker-nvidia", "reranker-voyageai"
]

TEXT2COLBERT_MODULES = ["text2colbert-jinaai"]

TEXT2VEC_MODULES = [
    "text2vec-aws", "text2vec-cohere", "text2vec-databricks", "text2vec-huggingface",
    "text2vec-jinaai", "text2vec-mistral", "text2vec-nvidia", "text2vec-octoai",
    "text2vec-ollama", "text2vec-openai", "text2vec-palm", "text2vec-voyageai", "text2vec-weaviate"
]


class WeaviateExperiment:
    """Class to experiment with Weaviate modules and their capabilities."""
    
    def __init__(self):
        """Initialize the Weaviate client and experiment setup."""
        self.client = None
        self.class_name = "ModuleTest"
        self.connect_to_weaviate()
        
    def connect_to_weaviate(self) -> None:
        """Connect to Weaviate Cloud."""
        try:
            logger.info(f"Connecting to Weaviate at {weaviate_url}")
            
            # Connect to Weaviate using the client API with v4 syntax
            self.client = weaviate.connect_to_weaviate_cloud(
                cluster_url=weaviate_url,
                auth_credentials=AuthApiKey(api_key=weaviate_api_key),
                headers={
                    "X-OpenAI-Api-Key": openai_api_key  # For OpenAI modules
                }
            )
            
            # Check if client is ready
            is_ready = self.client.is_ready()
            logger.info(f"Connection status: {is_ready}")
            
            if not is_ready:
                logger.error("Weaviate client is not ready")
                sys.exit(1)
                
        except Exception as e:
            logger.error(f"Error connecting to Weaviate: {e}")
            sys.exit(1)
    
    def close_connection(self) -> None:
        """Close the Weaviate client connection."""
        if self.client:
            logger.info("Closing Weaviate connection")
            self.client.close()
    
    def get_available_modules(self) -> Dict[str, List[str]]:
        """Get all available modules from the Weaviate instance."""
        try:
            logger.info("Retrieving available modules from Weaviate")
            
            # Get meta information using the client API
            meta = self.client.get_meta()
            
            # Extract modules from meta response
            modules = []
            if 'modules' in meta:
                modules = list(meta['modules'].keys())
                logger.info(f"Found {len(modules)} modules: {', '.join(modules)}")
            else:
                logger.warning("No modules found in Weaviate meta information, using predefined modules")
                # Use the predefined modules from the top of the file
                logger.info(f"Using predefined modules")
                
                # Return predefined module groups
                return {
                    "generative": GENERATIVE_MODULES,
                    "multi2vec": MULTI2VEC_MODULES,
                    "qna": QNA_MODULES,
                    "ref2vec": REF2VEC_MODULES,
                    "reranker": RERANKER_MODULES,
                    "text2colbert": TEXT2COLBERT_MODULES,
                    "text2vec": TEXT2VEC_MODULES,
                    "other": []
                }
            
            # Group modules by type
            module_groups = {
                "generative": [],
                "multi2vec": [],
                "qna": [],
                "ref2vec": [],
                "reranker": [],
                "text2colbert": [],
                "text2vec": [],
                "other": []
            }
            
            for module_name in modules:
                prefix = module_name.split('-')[0] if '-' in module_name else "other"
                if prefix in module_groups:
                    module_groups[prefix].append(module_name)
                else:
                    module_groups["other"].append(module_name)
            
            return module_groups
        except Exception as e:
            logger.error(f"Error retrieving modules: {e}")
            # Return predefined module groups as fallback
            return {
                "generative": GENERATIVE_MODULES[:2],  # Just use a couple for testing
                "text2vec": TEXT2VEC_MODULES[:2]     # Just use a couple for testing
            }
    
    def create_test_schema(self, vectorizer: str) -> None:
        """Create a test schema with the specified vectorizer."""
        try:
            # Check if class already exists and delete it
            schema = self.client.schema.get()
            classes = [cls['class'] for cls in schema['classes']] if 'classes' in schema else []
            
            if self.class_name in classes:
                logger.info(f"Deleting existing class: {self.class_name}")
                self.client.schema.delete_class(self.class_name)
            
            # Create class with specified vectorizer
            logger.info(f"Creating schema with vectorizer: {vectorizer}")
            
            # Create class definition
            class_obj = {
                "class": self.class_name,
                "description": f"Test class for {vectorizer}",
                "vectorizer": vectorizer,
                "properties": [
                    {
                        "name": "title",
                        "dataType": ["text"],
                        "description": "Title of the document"
                    },
                    {
                        "name": "content",
                        "dataType": ["text"],
                        "description": "Content of the document"
                    }
                ]
            }
            
            # Create the class
            self.client.schema.create_class(class_obj)
            logger.info(f"Schema created successfully: {self.class_name}")
            
        except Exception as e:
            logger.error(f"Error creating schema: {e}")
    
    def add_test_data(self) -> None:
        """Add test data to the collection."""
        try:
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
            
            logger.info(f"Adding {len(test_data)} test objects to {self.class_name}")
            
            # Use batch processing to add data
            with self.client.batch as batch:
                batch.batch_size = 10
                
                for idx, data in enumerate(test_data):
                    logger.info(f"Adding object {idx+1}: {data['title']}")
                    
                    # Generate a UUID for the object
                    object_uuid = str(uuid.uuid4())
                    
                    # Add the data object to the batch
                    batch.add_data_object(
                        data_object=data,
                        class_name=self.class_name,
                        uuid=object_uuid
                    )
            
            logger.info("Test data added successfully")
            
        except Exception as e:
            logger.error(f"Error adding test data: {e}")
    
    def test_vectorizer(self, vectorizer: str) -> None:
        """Test a specific vectorizer module."""
        try:
            logger.info(f"Testing vectorizer: {vectorizer}")
            
            # Create schema with this vectorizer
            self.create_test_schema(vectorizer)
            
            # Add test data
            self.add_test_data()
            
            # Perform a vector search
            query = "What is AI?"
            logger.info(f"Performing vector search with query: '{query}'")
            
            # Use GraphQL to perform a nearText search
            result = self.client.query.get(
                self.class_name, ["title", "content"]
            ).with_near_text({
                "concepts": [query]
            }).with_limit(2).do()
            
            # Process and log results
            if "data" in result and "Get" in result["data"]:
                objects = result["data"]["Get"][self.class_name]
                logger.info(f"Found {len(objects)} results")
                
                for idx, obj in enumerate(objects):
                    logger.info(f"Result {idx+1}: {obj['title']} - {obj['content'][:50]}...")
            else:
                logger.warning(f"No results found or unexpected response structure")
                logger.debug(f"Response: {json.dumps(result, indent=2)}")
            
        except Exception as e:
            logger.error(f"Error testing vectorizer {vectorizer}: {e}")
    
    def test_generative_module(self, module: str) -> None:
        """Test a generative module."""
        try:
            logger.info(f"Testing generative module: {module}")
            
            # Create schema with text2vec-openai as the vectorizer
            vectorizer = "text2vec-openai"
            self.create_test_schema(vectorizer)
            
            # Add test data
            self.add_test_data()
            
            # Perform a generative query
            logger.info(f"Performing generative query with {module}")
            
            # Set up generative parameters
            generative_config = {
                "singleResult": True
            }
            
            # Use GraphQL to perform a generative search
            result = self.client.query.get(
                self.class_name, ["title", "content"]
            ).with_generate(
                single_prompt="Summarize this text:",
                properties=["content"],
                grouped_task="summarize"
            ).with_limit(2).do()
            
            # Process and log results
            if "data" in result and "Get" in result["data"]:
                objects = result["data"]["Get"][self.class_name]
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
            logger.error(f"Error testing generative module {module}: {e}")
    
    def run_module_tests(self) -> None:
        """Run tests for different module types."""
        # Get available modules
        available_modules = self.get_available_modules()
        
        # Print available modules
        logger.info("Available modules by category:")
        for category, modules in available_modules.items():
            if modules:
                logger.info(f"  {category}: {', '.join(modules)}")
        
        # Test text vectorizers - just test one to save time
        # Prefer text2vec-openai if available, otherwise use the first one
        test_vectorizers = []
        if "text2vec" in available_modules and available_modules["text2vec"]:
            if "text2vec-openai" in available_modules["text2vec"]:
                test_vectorizers = ["text2vec-openai"]
            else:
                test_vectorizers = [available_modules["text2vec"][0]]
        
        for vectorizer in test_vectorizers:
            logger.info(f"\n{'='*50}\nTesting text vectorizer: {vectorizer}\n{'='*50}")
            self.test_vectorizer(vectorizer)
            time.sleep(1)  # Pause between tests
        
        # Test generative modules - just test one to save time
        # Prefer generative-openai if available, otherwise use the first one
        test_generative = []
        if "generative" in available_modules and available_modules["generative"]:
            if "generative-openai" in available_modules["generative"]:
                test_generative = ["generative-openai"]
            else:
                test_generative = [available_modules["generative"][0]]
        
        for gen_module in test_generative:
            logger.info(f"\n{'='*50}\nTesting generative module: {gen_module}\n{'='*50}")
            self.test_generative_module(gen_module)
            time.sleep(1)  # Pause between tests
        
        logger.info("\nModule testing completed. To test more modules, edit the run_module_tests method.")


def main():
    """Main function to run the Weaviate module experiments."""
    logger.info("Starting Weaviate module experiments")
    
    experiment = None
    
    try:
        # Create experiment instance
        experiment = WeaviateExperiment()
        
        # Run module tests
        experiment.run_module_tests()
        
    except KeyboardInterrupt:
        logger.info("Experiment interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
    finally:
        # Clean up
        if experiment:
            experiment.close_connection()
        logger.info("Weaviate module experiments completed")


if __name__ == "__main__":
    main()
