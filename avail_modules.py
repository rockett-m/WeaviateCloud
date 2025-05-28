# 38 modules active
# This file lists available Weaviate modules

# Generative modules
GENERATIVE_MODULES = [
    "generative-anthropic",
    "generative-anyscale",
    "generative-aws",
    "generative-cohere",
    "generative-databricks",
    "generative-friendliai",
    "generative-mistral",
    "generative-nvidia",
    "generative-octoai",
    "generative-ollama",
    "generative-openai",
    "generative-palm",
    "generative-xai"
]

# Multi2vec modules
MULTI2VEC_MODULES = [
    "multi2vec-cohere",
    "multi2vec-jinaai",
    "multi2vec-nvidia",
    "multi2vec-palm",
    "multi2vec-voyageai"
]

# QnA modules
QNA_MODULES = [
    "qna-openai"
]

# Ref2vec modules
REF2VEC_MODULES = [
    "ref2vec-centroid"
]

# Reranker modules
RERANKER_MODULES = [
    "reranker-cohere",
    "reranker-jinaai",
    "reranker-nvidia",
    "reranker-voyageai"
]

# Text2colbert modules
TEXT2COLBERT_MODULES = [
    "text2colbert-jinaai"
]

# Text2vec modules
TEXT2VEC_MODULES = [
    "text2vec-aws",
    "text2vec-cohere",
    "text2vec-databricks",
    "text2vec-huggingface",
    "text2vec-jinaai",
    "text2vec-mistral",
    "text2vec-nvidia",
    "text2vec-octoai",
    "text2vec-ollama",
    "text2vec-openai",
    "text2vec-palm",
    "text2vec-voyageai",
    "text2vec-weaviate"
]

# All modules combined
ALL_MODULES = (
    GENERATIVE_MODULES + 
    MULTI2VEC_MODULES + 
    QNA_MODULES + 
    REF2VEC_MODULES + 
    RERANKER_MODULES + 
    TEXT2COLBERT_MODULES + 
    TEXT2VEC_MODULES
)
