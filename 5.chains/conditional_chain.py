from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnablePassthrough,
    Runnable,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from typing_extensions import Literal

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-flash-lite-latest", temperature=0)
parser = StrOutputParser()


class Feedback(BaseModel):
  sentiment: Literal["positive", "negative"] = Field(
      description="Give the sentiment of the feedback"
  )


parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template=(
        "Classify the sentiment of the following feedback text into positive or"
        " negative \n {feedback} \n {format_instruction}"
    ),
    input_variables=["feedback"],
    partial_variables={"format_instruction": parser2.get_format_instructions()},
)

classifier_chain = prompt1 | model | parser2

# Step 1: Combine the original input dictionary with the parsed sentiment
classification_step = RunnablePassthrough.assign(sentiment_obj=classifier_chain)

prompt2 = PromptTemplate(
    template=(
        "Write an appropriate response to this positive feedback \n {feedback}"
    ),
    input_variables=["feedback"],
)

prompt3 = PromptTemplate(
    template=(
        "Write an appropriate response to this negative feedback \n {feedback}"
    ),
    input_variables=["feedback"],
)

# Step 2: Route using x["sentiment_obj"], but pass the full dict x (which contains 'feedback') to prompt2/prompt3
branch_chain = RunnableBranch(
    (
        lambda x: x["sentiment_obj"].sentiment == "positive",
        prompt2 | model | parser,
    ),
    (
        lambda x: x["sentiment_obj"].sentiment == "negative",
        prompt3 | model | parser,
    ),
    RunnableLambda(lambda x: "no sentiment"),
)

chain = classification_step | branch_chain

print(chain.invoke({"feedback": "this phone is bad, display not working"}))