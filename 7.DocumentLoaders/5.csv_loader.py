from langchain_community.document_loaders import CSVLoader

loader = CSVLoader('0_Social_Network_Ads.csv')
docs = loader.load()

print(len(docs))

print(docs[0])