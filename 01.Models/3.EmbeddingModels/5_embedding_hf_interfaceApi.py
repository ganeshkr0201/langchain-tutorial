from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv

load_dotenv()

document = [
    "Delhi is the capital of india",
    "Kolkata is the capital of West Bengal",
    "Paris is the capital of France"
]

embeddingModel = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2"
)


vectorEmbedding = embeddingModel.embed_documents(document)

print(vectorEmbedding)