from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='google/gemma-2-2b-it',
    task='text-generation'
)

model = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(
    template='tell me a joke about {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='explain me about this joke - {joke}',
    input_variables=['joke']
)

parser = StrOutputParser()

joke_gen_chain = RunnableSequence(prompt1, model, parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'explanation': RunnableSequence(prompt2, model, parser)
})

chain = RunnableSequence(joke_gen_chain, parallel_chain)

result = chain.invoke({'topic': 'ai'})
print(result)

chain.get_graph().print_ascii()