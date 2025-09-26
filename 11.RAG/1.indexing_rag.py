from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_chroma import Chroma

load_dotenv()


video_id = "Gfr50f6ZBvo"
lang = "en"

try: 
    # If you don’t care which language, this returns the “best” one
    yt_transcript = YouTubeTranscriptApi()
    transcript_list = yt_transcript.fetch(video_id, languages=[lang]).snippets

    # Flatten it to plain text
    transcript = " ".join(chunk.text for chunk in transcript_list) 

except Exception as e:
    print(str(e))


# creating a Text Splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# splitting text and creating documents
chunks = text_splitter.create_documents([transcript])


# initialising an embedding model
embeddings = HuggingFaceEndpointEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

# creating a vector store 
vector_store = Chroma(
    collection_name="youtube_transcript_rag",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db"
)

# adding documents to vector store
vector_store.add_documents(chunks)