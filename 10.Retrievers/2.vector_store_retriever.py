import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document


load_dotenv()

# creating vector store
vector_store = Chroma(
    collection_name="sample",
    embedding_function=GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001"),
    chroma_cloud_api_key=os.getenv("CHROMA_API_KEY"),
    tenant=os.getenv("CHROMA_TENANT"),
    database=os.getenv("CHROMA_DATABASE")
)


# convert vector store into retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

query = "who is rohit sharma?"

result = retriever.invoke(query)

for i, doc in enumerate(result):
    print(f"\n--- DOC {i+1} ---")
    print(doc.page_content)
