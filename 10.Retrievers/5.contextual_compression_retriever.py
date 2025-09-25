import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

load_dotenv()

# initializing llm model
llm_model = ChatGoogleGenerativeAI(model='gemini-1.5-flash')


# initializing embedding model 
embedding_model = GoogleGenerativeAIEmbeddings(model='models/gemini-embedding-001')


# creating vector store
vector_store = PineconeVectorStore(
    index_name=os.getenv("PINECONE_INDEX_NAME"),
    embedding=embedding_model
)


# creating contextual compression retriever
ccr_retriver = ContextualCompressionRetriever(
    base_retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
    base_compressor=LLMChainExtractor.from_llm(llm_model)
)


query = "What is photosynthesis?"

compressed_result = ccr_retriver.invoke(query)


for i, res in enumerate(compressed_result):
    print(f"--- Result {i+1} ----")
    print(res.page_content)