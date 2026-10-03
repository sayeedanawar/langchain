from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableSequence,RunnablePassthrough


from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
model = ChatGoogleGenerativeAI(model='gemini-2.5-flash', temperature=1.5, max_output_tokens=10000)
parser = StrOutputParser()


prompt1 = PromptTemplate(
    template='Generate a tweet about {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a Linkedin post about {topic}',
    input_variables=['topic']
)




# parallel_chain = RunnableParallel({
#     'tweet' : RunnableSequence(prompt1, model, parser),
#     'linkedin': prompt2 | model | parser
# })


parallel_chain = RunnablePassthrough() | {
    "tweet": prompt1 | model | parser,
    "linkedin": prompt2 | model | parser
}

result = parallel_chain.invoke({'topic':'when AGI will kill human?'})

print(result.get('tweet'))