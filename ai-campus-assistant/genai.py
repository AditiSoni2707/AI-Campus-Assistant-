from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory

# ---------------------------
# LLM
# ---------------------------
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# ---------------------------
# TOOLS
# ---------------------------
def campus_rules_tool(query: str) -> str:
    return (
        "Campus Rules:\n"
        "- Students must carry ID cards\n"
        "- No mobile phones during exams\n"
        "- Maintain silence in library"
    )

def campus_schedule_tool(query: str) -> str:
    return (
        "Campus Schedule:\n"
        "- Library: 8 AM to 10 PM\n"
        "- Cafeteria: Opens at 7 AM\n"
        "- Final exams: December\n"
        "- Midterms: August"
    )

tools = [
    Tool(
        name="CampusRules",
        func=campus_rules_tool,
        description="Use this tool to answer questions about campus rules"
    ),
    Tool(
        name="CampusSchedule",
        func=campus_schedule_tool,
        description="Use this tool to answer questions about exam dates and timings"
    )
]

# ---------------------------
# MEMORY (SHORT-TERM CONTEXT)
# ---------------------------
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# ---------------------------
# AGENT (AUTONOMOUS)
# ---------------------------
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)



# ---------------------------
# CHAT LOOP
# ---------------------------
while True:
    user_input = input("Student: ")
    if user_input.lower() == "exit":
        break

    response = agent.run(user_input)
    print("Assistant:", response)
