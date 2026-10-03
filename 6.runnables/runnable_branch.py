from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableBranch


from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
model = ChatGoogleGenerativeAI(model='gemini-2.5-flash', temperature=1.5, max_output_tokens=10000)
parser = StrOutputParser()


prompt1 = PromptTemplate(
    template='Write a details report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='summaize the following text \n {text}',
    input_variables=['text']
)

report_gen_chain = prompt1 | model | parser

branch_chain = RunnableBranch(
    (lambda report : len(report.split())>500, prompt2 | model | parser),
    RunnablePassthrough()
)

chain = report_gen_chain | branch_chain

# result = chain.invoke({'topic': 'forex market'})

# print(result)

chain.get_graph().print_ascii()
