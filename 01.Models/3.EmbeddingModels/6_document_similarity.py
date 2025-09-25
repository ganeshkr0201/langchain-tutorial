from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


load_dotenv()


documents = [
    "virat kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a fromer Indian captain famous for his demeanor and finishing skills.",
    "Sachin Tendulkar, also known as 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler know for his unorthodox action and yorkers."
]

embedding = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
)

query = 'tell me about virat kohli'



doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)



scores = cosine_similarity([query_embedding], doc_embeddings)[0]


index, score = (sorted(list(enumerate(scores)), key=lambda x:x[1])[-1]) 


print(query)
print(documents[index])
print("similarity score is : ", score)


