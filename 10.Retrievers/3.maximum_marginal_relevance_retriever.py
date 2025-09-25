import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain.schema import Document


load_dotenv()

# initializing embedding models
embedding = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")


# creating vector stores
vector_store = PineconeVectorStore(
    index_name=os.getenv("PINECONE_INDEX_NAME"),
    embedding=embedding
)


# convert vector store into retriever
retriever = vector_store.as_retriever(
    search_type="mmr", 
    search_kwargs={"k": 3, "lambda_mult": 0.5}
)


query = "What is langchain?"
result = retriever.invoke(query)

for i, doc in enumerate(result):
    print(f"\n--- DOC {i+1} ---")
    print(doc.page_content)