#!/usr/bin/env python3

import os
import sys
import time
import uuid
import logging
import json
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
import weaviate
from weaviate.auth import AuthApiKey

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("openai_modules_demo.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("OpenAIModulesDemo")

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


class OpenAIModulesDemo:
    """Class to demonstrate OpenAI modules in Weaviate."""
    
    def __init__(self):
        """Initialize the demo."""
        self.client = None
        self.class_name = "AIArticles"
        self.connect_to_weaviate()
    
    def connect_to_weaviate(self) -> None:
        """Connect to Weaviate Cloud."""
        try:
            logger.info(f"Connecting to Weaviate at {weaviate_url}")
            
            # Connect to Weaviate using the client API
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
    
    def setup_schema(self) -> None:
        """Set up the schema for the demo."""
        try:
            # Use collections API to create and manage schema
            logger.info(f"Creating schema with text2vec-openai vectorizer")
            
            # First, check if the collection exists and delete it if it does
            try:
                # Try to get the collection to see if it exists
                response = self.client._connection.get(f"collections/{self.class_name}")
                
                if response.status_code == 200:
                    # Collection exists, delete it
                    logger.info(f"Deleting existing collection: {self.class_name}")
                    delete_response = self.client._connection.delete(f"collections/{self.class_name}")
                    
                    if delete_response.status_code != 200:
                        logger.warning(f"Failed to delete collection: {delete_response.status_code} - {delete_response.text}")
                        time.sleep(1)  # Wait a bit before trying to create
            except Exception as e:
                # Collection might not exist, which is fine
                logger.info(f"Collection check failed, assuming it doesn't exist: {e}")
            
            # Create collection with text2vec-openai vectorizer
            collection_obj = {
                "name": self.class_name,
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
                    },
                    {
                        "name": "url",
                        "dataType": ["text"],
                        "description": "URL of the article",
                        "moduleConfig": {
                            "text2vec-openai": {
                                "skip": True
                            }
                        }
                    }
                ]
            }
            
            # Create the collection using direct REST API
            response = self.client._connection.post("collections", collection_obj)
            
            if response.status_code == 200:
                logger.info(f"Schema created successfully: {self.class_name}")
            else:
                logger.error(f"Failed to create schema: {response.status_code} - {response.text}")
                sys.exit(1)
            
        except Exception as e:
            logger.error(f"Error setting up schema: {e}")
            sys.exit(1)
    
    def add_sample_data(self) -> None:
        """Add sample data to the collection."""
        try:
            sample_data = [
                {
                    "title": "Understanding Large Language Models",
                    "content": "Large Language Models (LLMs) are deep learning algorithms that can recognize, summarize, translate, predict, and generate text and other content based on knowledge gained from massive datasets. They use transformer architectures to handle sequential data and attention mechanisms to focus on relevant parts of the input. Popular examples include GPT-4, Claude, and Gemini.",
                    "category": "AI Technology",
                    "url": "https://example.com/llm-overview"
                },
                {
                    "title": "Vector Databases for AI Applications",
                    "content": "Vector databases are specialized database systems designed to store and query high-dimensional vectors efficiently. They are crucial for AI applications that rely on embeddings, such as semantic search, recommendation systems, and similarity matching. Unlike traditional databases, vector databases use approximate nearest neighbor (ANN) algorithms to find similar vectors quickly.",
                    "category": "AI Infrastructure",
                    "url": "https://example.com/vector-databases"
                },
                {
                    "title": "The Ethics of Artificial Intelligence",
                    "content": "AI ethics involves designing and using AI systems responsibly and ethically. Key concerns include bias and fairness, privacy, transparency, accountability, and the long-term impacts on society and employment. As AI becomes more powerful and widespread, ensuring these systems align with human values and benefit humanity becomes increasingly important.",
                    "category": "AI Ethics",
                    "url": "https://example.com/ai-ethics"
                },
                {
                    "title": "Prompt Engineering Techniques",
                    "content": "Prompt engineering is the practice of designing effective prompts to get the best results from language models. Techniques include using clear instructions, providing examples (few-shot learning), breaking complex tasks into steps, and using specific formatting. Good prompt engineering can significantly improve the quality, relevance, and accuracy of AI-generated outputs.",
                    "category": "AI Usage",
                    "url": "https://example.com/prompt-engineering"
                },
                {
                    "title": "Multimodal AI Systems",
                    "content": "Multimodal AI systems can process and generate content across multiple types of data, such as text, images, audio, and video. These systems combine different neural network architectures to understand relationships between different modalities. Examples include GPT-4V, Gemini, and Claude Opus, which can analyze images and respond with text.",
                    "category": "AI Technology",
                    "url": "https://example.com/multimodal-ai"
                }
            ]
            
            logger.info(f"Adding {len(sample_data)} sample articles to {self.class_name}")
            
            # Add data using batch processing
            with self.client.batch as batch:
                batch.batch_size = 10
                
                for idx, data in enumerate(sample_data):
                    logger.info(f"Adding article {idx+1}: {data['title']}")
                    
                    # Generate a UUID for the object
                    object_uuid = str(uuid.uuid4())
                    
                    # Add the data object to the batch
                    batch.add_data_object(
                        data_object=data,
                        class_name=self.class_name,
                        uuid=object_uuid
                    )
            
            logger.info("Sample data added successfully")
            
            # Wait for indexing to complete
            logger.info("Waiting for indexing to complete...")
            time.sleep(2)
            
        except Exception as e:
            logger.error(f"Error adding sample data: {e}")
    
    def demonstrate_vector_search(self) -> None:
        """Demonstrate vector search using text2vec-openai."""
        try:
            logger.info("\n=== DEMONSTRATING VECTOR SEARCH WITH TEXT2VEC-OPENAI ===\n")
            
            # Perform a semantic search
            query = "What are the ethical considerations in AI?"
            logger.info(f"Performing semantic search with query: '{query}'")
            
            # Use the query API to perform a nearText search
            result = self.client._connection.post(
                "graphql",
                {
                    "query": f"""
                    {{
                      Get {{
                        {self.class_name}(
                          nearText: {{
                            concepts: ["{query}"]
                            certainty: 0.7
                          }}
                          limit: 3
                        ) {{
                          title
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
            
            if result.status_code == 200:
                data = result.json()
                
                if "data" in data and "Get" in data["data"] and self.class_name in data["data"]["Get"]:
                    articles = data["data"]["Get"][self.class_name]
                    logger.info(f"Found {len(articles)} relevant articles:")
                    
                    for idx, article in enumerate(articles):
                        certainty = article["_additional"]["certainty"] if "_additional" in article and "certainty" in article["_additional"] else "N/A"
                        logger.info(f"  {idx+1}. {article['title']} (Category: {article['category']}, Relevance: {certainty:.2f})")
                else:
                    logger.warning("No results found or unexpected response structure")
            else:
                logger.error(f"Search failed: {result.status_code} - {result.text}")
            
        except Exception as e:
            logger.error(f"Error demonstrating vector search: {e}")
    
    def demonstrate_generative_ai(self) -> None:
        """Demonstrate generative AI using generative-openai."""
        try:
            logger.info("\n=== DEMONSTRATING GENERATIVE AI WITH GENERATIVE-OPENAI ===\n")
            
            # Perform a generative query
            query = "What are vector databases?"
            logger.info(f"Performing generative search with query: '{query}'")
            
            # Use the query API to perform a search with generate
            result = self.client._connection.post(
                "graphql",
                {
                    "query": f"""
                    {{
                      Get {{
                        {self.class_name}(
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
            
            if result.status_code == 200:
                data = result.json()
                
                if "data" in data and "Get" in data["data"] and self.class_name in data["data"]["Get"]:
                    articles = data["data"]["Get"][self.class_name]
                    
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
                logger.error(f"Generative query failed: {result.status_code} - {result.text}")
            
            # Demonstrate a standalone generative query (not tied to specific objects)
            logger.info("\n=== DEMONSTRATING STANDALONE GENERATIVE QUERY ===\n")
            
            prompt = "Explain the difference between vector databases and traditional databases in 3 bullet points"
            logger.info(f"Sending standalone prompt: '{prompt}'")
            
            # Use the generate API endpoint
            result = self.client._connection.post(
                "modules/generative-openai/generate",
                {
                    "prompt": prompt
                }
            )
            
            if result.status_code == 200:
                data = result.json()
                if "text" in data:
                    logger.info(f"Generated response:\n{data['text']}")
                else:
                    logger.warning("No generated text in response")
            else:
                logger.error(f"Standalone generation failed: {result.status_code} - {result.text}")
            
        except Exception as e:
            logger.error(f"Error demonstrating generative AI: {e}")
    
    def run_demo(self) -> None:
        """Run the complete OpenAI modules demonstration."""
        try:
            # Set up schema
            self.setup_schema()
            
            # Add sample data
            self.add_sample_data()
            
            # Demonstrate vector search
            self.demonstrate_vector_search()
            
            # Demonstrate generative AI
            self.demonstrate_generative_ai()
            
        except Exception as e:
            logger.error(f"Error running demo: {e}")


def main():
    """Main function to run the OpenAI modules demonstration."""
    logger.info("Starting OpenAI modules demonstration")
    
    demo = None
    
    try:
        # Create and run the demo
        demo = OpenAIModulesDemo()
        demo.run_demo()
        
    except KeyboardInterrupt:
        logger.info("Demonstration interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
    finally:
        # Clean up
        if demo:
            demo.close_connection()
        logger.info("OpenAI modules demonstration completed")


if __name__ == "__main__":
    main()
