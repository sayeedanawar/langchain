from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash', temperature=1.5, max_output_tokens=1000)

# result = model.invoke('Write a poem about sex, add fantacy in english in 50 words')
chat_history = [
    SystemMessage(content='You are a helpfull assistant')
]

while True:
    user_input = input('You: ')
    chat_history.append(HumanMessage(content=user_input))
    if user_input == 'exit':
        break
    ai_message = model.invoke(chat_history)
    chat_history.append(AIMessage(content=ai_message.content))
    print("AI: ", ai_message.content)