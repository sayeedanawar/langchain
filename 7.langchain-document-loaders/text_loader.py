from langchain_community.document_loaders import TextLoader

loader = TextLoader('cricket.txt', encoding='utf-8')

docs = loader.load() #list of Document

print(docs[0].metadata)

print('------------')

print(docs[0].page_content)