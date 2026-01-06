L
# 1. Imports

from langgraph.graph import StateGraph, MessagesState
from langgraph.checkpoint.memory import MemorySaver

from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage



# 2. Initialize LLM

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)



# 3. System Prompt

SYSTEM_PROMPT = SystemMessage(
    content=(
        "You are a helpful, polite customer support representative. "
        "Your job is to assist customers with product issues, ask clear "
        "follow-up questions, and provide simple troubleshooting steps."
    )
)



# 4. Agent Node

def customer_support_agent(state: MessagesState):
    messages = state["messages"]

    # Ensure system prompt is always included
    if not any(isinstance(msg, SystemMessage) for msg in messages):
        messages = [SYSTEM_PROMPT] + messages

    # Call the LLM
    response = llm.invoke(messages)

    # Return updated state
    return {
        "messages": messages + [response]
    }



# 5. Build StateGraph

graph = StateGraph(MessagesState)

# Add agent node
graph.add_node("support_agent", customer_support_agent)

# Set entry point
graph.set_entry_point("support_agent")


# 6. Add Memory (Checkpointer)

memory = MemorySaver()

# Compile app
app = graph.compile(checkpointer=memory)



# 7. Test Multi-Turn Conversation

thread_id = "customer-session-1"

# ---- Turn 1 ----
result = app.invoke(
    {
        "messages": [
            HumanMessage(content="I bought a laptop last week")
        ]
    },
    config={"configurable": {"thread_id": thread_id}}
)

print("Agent:", result["messages"][-1].content)


# ---- Turn 2 (context remembered) ----
result = app.invoke(
    {
        "messages": [
            HumanMessage(content="It won't turn on")
        ]
    },
    config={"configurable": {"thread_id": thread_id}}
)

print("Agent:", result["messages"][-1].content)


# ---- Turn 3 (still remembers laptop) ----
result = app.invoke(
    {
        "messages": [
            HumanMessage(content="Yes, I tried charging it overnight")
        ]
    },
    config={"configurable": {"thread_id": thread_id}}
)

print("Agent:", result["messages"][-1].content)
