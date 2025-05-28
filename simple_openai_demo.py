#!/usr/bin/env python3

import os
import sys
import time
import uuid
import logging
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
logger = logging.getLogger("SimpleOpenAIDemo")

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
    """Main function to demonstrate OpenAI modules in Weaviate."""
    logger.info("Starting simple OpenAI modules demonstration")
    
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
        
        # Get meta information to see available modules
        logger.info("Retrieving available modules")
        meta = client.get_meta()
        
        if 'modules' in meta:
            modules = list(meta['modules'].keys())
            logger.info(f"Available modules: {', '.join(modules)}")
            
            # Check if the required OpenAI modules are available
            if 'text2vec-openai' not in modules or 'generative-openai' not in modules:
                logger.warning("Required OpenAI modules not available")
                logger.warning("text2vec-openai: " + ("Available" if 'text2vec-openai' in modules else "Not available"))
                logger.warning("generative-openai: " + ("Available" if 'generative-openai' in modules else "Not available"))
        
        # List existing schema
        logger.info("Checking existing schema")
        try:
            schema = client.schema.get()
            existing_classes = [cls['class'] for cls in schema['classes']] if 'classes' in schema else []
            logger.info(f"Existing classes: {', '.join(existing_classes) if existing_classes else 'None'}")
            
            # Delete the class if it exists
            if CLASS_NAME in existing_classes:
                logger.info(f"Deleting existing class: {CLASS_NAME}")
                client.schema.delete_class(CLASS_NAME)
                logger.info(f"Class {CLASS_NAME} deleted")
        except Exception as e:
            logger.error(f"Error checking schema: {e}")
        
        # Create a new class with text2vec-openai vectorizer
        logger.info(f"Creating class {CLASS_NAME} with text2vec-openai vectorizer")
        try:
            class_obj = {
                "class": CLASS_NAME,
                "description": "AI-related articles for demonstration",
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
                        "description": "Title of the article"
                    },
                    {
                        "name": "content",
                        "dataType": ["text"],
                        "description": "Content of the article"
                    },
                    {
                        "name": "category",
                        "dataType": ["text"],
                        "description": "Category of the article"
                    }
                ]
            }
            
            client.schema.create_class(class_obj)
            logger.info(f"Class {CLASS_NAME} created successfully")
        except Exception as e:
            logger.error(f"Error creating class: {e}")
            sys.exit(1)
        
        # Add sample data
        logger.info("Adding sample data")
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
            
            # Add data using batch processing
            with client.batch as batch:
                batch.batch_size = 10
                
                for idx, data in enumerate(sample_data):
                    logger.info(f"Adding article {idx+1}: {data['title']}")
                    
                    # Generate a UUID for the object
                    object_uuid = str(uuid.uuid4())
                    
                    # Add the data object to the batch
                    batch.add_data_object(
                        data_object=data,
                        class_name=CLASS_NAME,
                        uuid=object_uuid
                    )
            
            logger.info("Sample data added successfully")
            
            # Wait for indexing to complete
            logger.info("Waiting for indexing to complete...")
            time.sleep(2)
        except Exception as e:
            logger.error(f"Error adding sample data: {e}")
        
        # Perform a vector search
        logger.info("\n=== DEMONSTRATING VECTOR SEARCH WITH TEXT2VEC-OPENAI ===\n")
        try:
            query = "What are the ethical considerations in AI?"
            logger.info(f"Performing semantic search with query: '{query}'")
            
            result = (
                client.query
                .get(CLASS_NAME, ["title", "content", "category"])
                .with_near_text({"concepts": [query]})
                .with_limit(2)
                .do()
            )
            
            if "data" in result and "Get" in result["data"]:
                articles = result["data"]["Get"][CLASS_NAME]
                logger.info(f"Found {len(articles)} relevant articles:")
                
                for idx, article in enumerate(articles):
                    logger.info(f"  {idx+1}. {article['title']} (Category: {article['category']})")
                    logger.info(f"     Content: {article['content'][:100]}...")
            else:
                logger.warning("No results found or unexpected response structure")
        except Exception as e:
            logger.error(f"Error performing vector search: {e}")
        
        # Perform a generative query
        logger.info("\n=== DEMONSTRATING GENERATIVE AI WITH GENERATIVE-OPENAI ===\n")
        try:
            query = "What are vector databases?"
            logger.info(f"Performing generative search with query: '{query}'")
            
            result = (
                client.query
                .get(CLASS_NAME, ["title", "content"])
                .with_near_text({"concepts": [query]})
                .with_generate({"singleResult": True, "prompt": "Explain this in simpler terms:"})
                .with_limit(1)
                .do()
            )
            
            if "data" in result and "Get" in result["data"]:
                articles = result["data"]["Get"][CLASS_NAME]
                
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
        except Exception as e:
            logger.error(f"Error performing generative search: {e}")
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
    finally:
        # Clean up
        if client:
            client.close()
        logger.info("OpenAI modules demonstration completed")


if __name__ == "__main__":
    main()
