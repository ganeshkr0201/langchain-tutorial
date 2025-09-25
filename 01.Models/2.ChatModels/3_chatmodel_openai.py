from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

model = ChatOpenAI(model="gpt-3.5-turbo-instruct", temperature=1.2, max_completion_tokens=20)
result = model.invoke("write a five lines poem on cricket")

print(result.content)