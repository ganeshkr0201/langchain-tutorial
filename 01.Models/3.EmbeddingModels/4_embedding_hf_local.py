from langchain_huggingface import HuggingFaceEmbeddings

document = [
    "Delhi is the capital of india",
    "Kolkata is the capital of West Bengal",
    "Paris is the capital of France"
]

embeddingModel = HuggingFaceEmbeddings(
    model='sentence-transformers/all-MiniLM-L6-v2'
)


result = embeddingModel.embed_documents(document)

print(result)