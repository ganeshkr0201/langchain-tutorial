from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableLambda

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='mistralai/Mixtral-8x7B-Instruct-v0.1',
    task='chat-completion'
)

model = ChatHuggingFace(llm=llm)

url = 'https://en.wikipedia.org/wiki/India_Gate'
loader = WebBaseLoader(url)
docs = loader.load()

parser = StrOutputParser()

prompt = PromptTemplate(
    template='Answer the following query {query} \n from the the following text - \n {document}',
    input_variables=['query', 'document']
)

web_loader_chain = RunnableLambda(lambda x: {'query': x['query'], 'document': loader.load()[0].page_content})

chain = web_loader_chain | prompt | model | parser

result = chain.invoke({'query': 'what is the height of india gate?'})

print(result)