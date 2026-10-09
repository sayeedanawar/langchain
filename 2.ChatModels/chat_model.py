from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", temperature=1.5, max_output_tokens=1000
)

result = model.invoke("Write a poem about AI in 50 words")

print(result)
print("test msg")
