#!/usr/bin/env python3

import os
import sys
from dotenv import load_dotenv
import weaviate
from weaviate.classes.init import Auth

# Get the project root directory
ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.append(ROOT)

# Best practice: store your credentials in environment variables
load_dotenv()
weaviate_url = os.getenv("WEAVIATE_URL", "")
weaviate_api_key = os.getenv("WEAVIATE_API_KEY", "")
rest_endpoint = os.getenv("REST_ENDPOINT", "")
gpc_endpoint = os.getenv("GRPC_ENDPOINT", "")
openai_api_key = os.getenv("OPENAI_API_KEY", "")

if len(weaviate_url) == 0 or len(weaviate_api_key) == 0 or len(rest_endpoint) == 0 or len(gpc_endpoint) == 0 or len(openai_api_key) == 0:
    raise ValueError("WEAVIATE_URL or WEAVIATE_API_KEY or REST_ENDPOINT or GRPC_ENDPOINT or OPENAI_API_KEY is not set")


def connect_weave_cloud() -> weaviate.Client | None:
    try:
        # Connect to Weaviate Cloud
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=weaviate_url,
            auth_credentials=Auth.api_key(weaviate_api_key),
        )
        print(f"{client.is_ready() = }")
        return client

    except Exception as e:
        print(f"Error connecting to Weaviate Cloud: {e}")
        return None


# hit the gpc endpoint
def test_gpc_endpoint() -> weaviate.Client | None:
    client = None
    try:
        # Connect to Weaviate Cloud
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=gpc_endpoint,
            auth_credentials=Auth.api_key(weaviate_api_key),
        )
        print(f"{client.is_ready() = }")
        return client

    except Exception as e:
        print(f"Error connecting to Weaviate Cloud: {e}")
        if client:
            client.close()
        return None

# hit the weaviate endpoint
def test_weaviate_endpoint() -> weaviate.Client | None:
    client = None
    try:
        # Connect to Weaviate Cloud
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=rest_endpoint,
            auth_credentials=Auth.api_key(weaviate_api_key),
        )
        print(f"{client.is_ready() = }")
        return client

    except Exception as e:
        print(f"Error connecting to Weaviate Cloud: {e}")
        if client:
            client.close()
        return None

# get some data from weaviate
def get_data_from_weaviate() -> weaviate.Client | None:
    client = None
    try:
        # Connect to Weaviate Cloud
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=rest_endpoint,
            auth_credentials=Auth.api_key(weaviate_api_key),
        )
        print(f"{client.is_ready() = }")
        return client

    except Exception as e:
        print(f"Error connecting to Weaviate Cloud: {e}")
        if client:
            client.close()
        return None

# get some data from weaviate
def get_data_from_weaviate() -> weaviate.Client | None:
    client = None
    try:
        # Connect to Weaviate Cloud
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=rest_endpoint,
            auth_credentials=Auth.api_key(weaviate_api_key),
        )
        print(f"{client.is_ready() = }")
        return client

    except Exception as e:
        print(f"Error connecting to Weaviate Cloud: {e}")
        if client:
            client.close()
        return None




if __name__ == "__main__":
    # Connect to main Weaviate Cloud instance
    client = connect_weave_cloud()
    if client:
        client.close()
        print("Client closed")
    else:
        print("Client failed to connect")

    # Test other endpoints and make sure to close connections
    client = test_gpc_endpoint()
    if client:
        client.close()
        print("GPC endpoint client closed")

    client = test_weaviate_endpoint()
    if client:
        client.close()
        print("REST endpoint client closed")

    client = get_data_from_weaviate()
    if client:
        client.close()
        print("Data client 1 closed")

    client = get_data_from_weaviate()
    if client:
        client.close()
        print("Data client 2 closed")
