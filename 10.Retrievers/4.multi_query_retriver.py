import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain.retrievers import MultiQueryRetriever


load_dotenv()

# initialising llm model
llm_model = ChatGoogleGenerativeAI(model='gemini-1.5-flash')

# initialising embedding model
embedding_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")


# creating vector_stroe
vector_store = PineconeVectorStore(
    index_name=os.getenv("PINECONE_INDEX_NAME"),
    embedding=embedding_model,
)


# creating multiquery retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 4})

multiquery_retriever = MultiQueryRetriever.from_llm(
    llm=llm_model,
    retriever=retriever,
)


# Retriving Result form vector database
query = "How to improve energy levels and maintain balance?"
multiquery_result = multiquery_retriever.invoke(query)


# Printing Results
for i, res in enumerate(multiquery_result):
    print(f"--- Result {i+1} ---")
    print(res.page_content)