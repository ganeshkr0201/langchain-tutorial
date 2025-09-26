from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# initialising google chat model
llm = HuggingFaceEndpoint(
    repo_id='google/gemma-2-2b-it',
    task='text-generation'
)

model = ChatHuggingFace(llm=llm)

# initialising embedding model
embeddings = HuggingFaceEndpointEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

# creating a parser
parser = StrOutputParser()

# creating connection with vector store
vector_store = Chroma(
    collection_name="youtube_transcript_rag",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db"
)


# creating a retriever
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})


# creating a prompt template
prompt = PromptTemplate(
    template="""
            You are a helpful assistant,
            Answer ONLY from the provided transcript context.
            If the context is insufficient, just say you don't know.

            {context}
            Question: {question}
        """,
        input_variables=['context', 'question']
)

question = "is the topis of aliens disscussed in this video? if yes then what was discussed"


# creating a parallel chain which send question and formated context into the main chain
parallel_chain = RunnableParallel({
    'question': RunnablePassthrough(),
    'context': retriever | RunnableLambda(lambda format_docs: "\n\n".join(doc.page_content for doc in format_docs))
})

# creating a series chain which sends prompt to llm model and then parser to get the result
main_chain = parallel_chain | prompt | model | parser

# invoking main chain to get the result
result = main_chain.invoke(question)
print(result)


# printing the graph of the chain diagram
main_chain.get_graph().print_ascii()














# # retrieving documents from the vector store and converting into string
# retriever_docs = retriever.invoke(question)
# context_text = " ".join(docs.page_content for docs in retriever_docs)


# # creating final prompt
# final_prompt = prompt.invoke({"question": question, "context": context_text}) 

# # sending requestion to llm 
# response = model.invoke(final_prompt)
# print(response.content)