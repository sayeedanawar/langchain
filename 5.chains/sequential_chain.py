from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser



from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
model = ChatGoogleGenerativeAI(model='gemini-2.5-flash', temperature=1.5, max_output_tokens=10000)




prompt1 = PromptTemplate(
    template='Gererate detailed report about {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Gererate a 5 pointer summery from the follwoing text \n {text}',
    input_variables=['text']
)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser
chain.get_graph().print_ascii()


result = chain.invoke({'topic':'yeasterdays indian stock market crash'})

print(result)


