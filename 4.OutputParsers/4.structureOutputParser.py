from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)


schema = [
    ResponseSchema(name='fact_1', description="fact 1 about the topic"),
    ResponseSchema(name='fact_2', description='fact 2 about the topic'),
    ResponseSchema(name='fact_3', description='fact 3 about the topic'),
    ResponseSchema(name='fact_4', description="fact 4 about the topic"),
    ResponseSchema(name='fact_5', description='fact 5 about the topic'),
    ResponseSchema(name='fact_6', description='fact 6 about the topic'),
    ResponseSchema(name='fact_7', description="fact 7 about the topic"),
    ResponseSchema(name='fact_8', description='fact 8 about the topic'),
    ResponseSchema(name='fact_9', description='fact 9 about the topic'),
    ResponseSchema(name='fact_10', description='fact 10 about the topic')
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template="Give 10 facts about {topic} \n {format_instruction}",
    input_variables=['topic'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({'topic': 'black hole'})

print(result)



