from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('0_dl-curriculum.pdf')

docs = loader.load()

print(docs[0])
print(docs)