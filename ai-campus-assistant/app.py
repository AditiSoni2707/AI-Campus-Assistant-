from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(model="gpt-3.5-turbo")

print("LEVEL 1: Basic Campus Assistant (type exit to quit)")

while True:
    query = input("Student: ")
    if query.lower() == "exit":
        break

    response = llm.invoke(query)
    print("Assistant:", response.content)

