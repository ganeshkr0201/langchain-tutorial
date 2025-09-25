from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from langchain.schema.runnable import RunnableBranch, RunnableLambda, RunnableMap


load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

llm = HuggingFaceEndpoint(
    repo_id='google/gemma-2-2b-it',
    task='text-generation'
)

model2 = ChatHuggingFace(llm=llm)

class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description="Give the sentiment of the feedback")


string_output_parser = StrOutputParser()
pydantic_output_parser = PydanticOutputParser(pydantic_object=Feedback)


prompt1 = PromptTemplate(
    template='Classify the sentiment of the following feedback text into positive or negative \n {feedback} \n {format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction':pydantic_output_parser.get_format_instructions()}
)

prompt2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)


classifier_chain = prompt1 | model | pydantic_output_parser

classifier_with_input = RunnableMap({
    "sentiment": classifier_chain,
    "feedback": lambda x: x['feedback']
})

branch_chain = RunnableBranch(
    (lambda x: x['sentiment'].sentiment == 'positive', prompt2 | model | string_output_parser),
    (lambda x: x['sentiment'].sentiment == 'negative', prompt3 | model | string_output_parser),
    RunnableLambda(lambda x: 'could not find sentiment')
)

chain = classifier_with_input | branch_chain

result = chain.invoke({'feedback': 'this phone is beautiful'})

print(result)

chain.get_graph().print_ascii()