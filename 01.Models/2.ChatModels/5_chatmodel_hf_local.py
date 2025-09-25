from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    temperature=0.5,
    max_new_tokens=50,
)

# Wrap with LangChain
llm = HuggingFacePipeline(pipeline=pipe)

# Chat wrapper
model = ChatHuggingFace(llm=llm)

# Run query
result = model.invoke("what is the capital of india?")
print(result)
