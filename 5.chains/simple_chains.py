from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser



from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
model = ChatGoogleGenerativeAI(model='gemini-2.5-flash', temperature=1.5, max_output_tokens=10000)

prompt = PromptTemplate(
    template='Gererate 5 interesting facts about {topic}',
    input_variables=['topic']
)

parser = StrOutputParser()

chain = prompt | model | parser
chain.get_graph().print_ascii()


result = chain.invoke({'topic':'forex trading'})

print(result)


