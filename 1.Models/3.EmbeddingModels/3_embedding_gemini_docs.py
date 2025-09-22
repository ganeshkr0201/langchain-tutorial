from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

document = [
    "Delhi is the capital of india",
    "Kolkata is the capital of West Bengal",
    "Paris is the capital of France"
]

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

vector = embeddings.embed_documents(document)

print(vector)

