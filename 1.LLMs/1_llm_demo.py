from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
 
# Initialize Gemini
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Invoke the model
response = llm.invoke("What is the capital of India")

# Chat models return an AIMessage object, so access .content for text
print(response.content)