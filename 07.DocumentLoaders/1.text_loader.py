from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableLambda

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='google/gemma-2-2b-it',
    task='text-generation'
)

model = ChatHuggingFace(llm=llm)

loader = TextLoader('0_cricket.txt')

parser = StrOutputParser()

prompt = PromptTemplate(
    template='Summarize this poem \n {poem}',
    input_variables=['poem']
)

loader_chain = RunnableLambda(lambda _: loader.load()[0].page_content)

chain = loader_chain | prompt | model | parser

result = chain.invoke({})
print(result)
