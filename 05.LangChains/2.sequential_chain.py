from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='google/gemma-2-2b-it',
    task='text-generation'
)

model = ChatHuggingFace(llm=llm)

template1 = PromptTemplate(
    template='generate a detailed report on the topic {topic}',
    input_variables=['topic']
)

template2 = PromptTemplate(
    template='give me the five most important point from the content {content}',
    input_variables=['content']
)

parser = StrOutputParser()


chain = template1 | model | parser | template2 | model | parser
result = chain.invoke({'topic': 'unemployment in india'})

print(result)

chain.get_graph().print_ascii()