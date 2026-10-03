from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableSequence, RunnablePassthrough, RunnableLambda


from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
model = ChatGoogleGenerativeAI(model='gemini-2.5-flash', temperature=1.5, max_output_tokens=10000)
parser = StrOutputParser()


prompt1 = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)


joke_gen_chain = RunnableSequence(prompt1, model, parser)



parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'count': RunnableLambda(lambda joke : len(joke.split()))
})

chain = joke_gen_chain | parallel_chain
result = chain.invoke({'topic':'Human'})

print(result)