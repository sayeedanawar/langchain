from langchain_community.document_loaders import PyPDFLoader


loader = PyPDFLoader('file-example_PDF_500_kB.pdf')


docs = loader.load()

print(docs[0].metadata)

print('------------')

print(docs[0].page_content)