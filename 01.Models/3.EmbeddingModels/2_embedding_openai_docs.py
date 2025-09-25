from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()


document = [
    "Delhi is the capital of india",
    "Kolkata is the capital of West Bengal",
    "Paris is the capital of France"
]

EmbeddingModel = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=32)

resultVector = EmbeddingModel.embed_documents(document)

print(str(resultVector))