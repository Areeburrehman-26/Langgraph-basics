#!.venv/Scripts/python.exe

from dotenv import load_dotenv
from typing import Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from groq import Groq
from openai import OpenAI
import os
from langchain_openai import ChatOpenAI
load_dotenv()


# ✅ Point ChatOpenAI to Groq’s OpenAI-compatible endpoint
llm = ChatOpenAI(
    model="openai/gpt-oss-120b",   # Groq model ID
    api_key=os.environ["GROQ_API_KEY"],  # use your Groq key, not OpenAI key
    base_url="https://api.groq.com/openai/v1",
    temperature=1,
)


class MessageClassifier(BaseModel):
    message_type: Literal["emotional", "Logical"] = Field(
        ...,
        description="Classify if the message requires an emotional or logical response.",
    )


class State(TypedDict):
    messages: Annotated[list, add_messages]
    message_type: str | None


def classify_message(state: State):
    last_msg = state["messages"][-1]
    classifier_llm = llm.with_structured_output(MessageClassifier)
    result = classifier_llm.invoke([
        {
            "role": "system",
            "content":"""classify the user message as either :
                - 'emotional': if the message requires an emotional response, such as Therapy, empathy, encouragement, or understanding.
                - 'Logical': if the message requires a logical response, such as facts, data, reasoning, or problem-solving.
            """
            
        },
        {
            "role": "user",
            "content": last_msg.content,
        }
    ])
    return {"message_type":result.message_type}

def router(state: State):
    message_type = state.get("message_type", "logical")
    if message_type == "emotional":
        return {"next": "therapist"}
    
    return {"next": "logical"}

def therapist_agent(state: State):
    last_message = state["messages"][-1]
    messages=[
        {"role":"system","content":"You are a compassionate and empathetic therapist. Your goal is to provide emotional support and understanding to the user."},
        {"role":"user","content":last_message.content}
    ]
    reply = llm.invoke(messages)
    return {"messages":[{"role":"assistant","content":reply.content}]}


def logical_agent(state: State):
    last_message = state["messages"][-1]
    messages=[
        {"role":"system","content":"You are a knowledgeable and logical assistant. Your goal is to provide clear, factual, and reasoned responses to the user."},
        {"role":"user","content":last_message.content}
    ]
    reply = llm.invoke(messages)
    return {"messages":[{"role":"assistant","content":reply.content}]}



graphbuilder =StateGraph(State)



graphbuilder.add_node("classifier", classify_message)
graphbuilder.add_node("router", router)
graphbuilder.add_node("therapist", therapist_agent)
graphbuilder.add_node("logical", logical_agent)

graphbuilder.add_edge(START, "classifier")
graphbuilder.add_edge("classifier", "router")

graphbuilder.add_conditional_edges(
     "router",
     lambda state: state.get("next"),
     {"therapist": "therapist", "logical": "logical"},
)
graphbuilder.add_edge("therapist", END)
graphbuilder.add_edge("logical", END) 


graph =  graphbuilder.compile()

def run_chatbot():
    state = {"messages": [], "message_type": None}

    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting chat...")
            break

        state["messages"] = state.get("messages", []) + [{"role": "user", "content": user_input}]

        state= graph.invoke(state)

        if state.get("messages") and len(state["messages"]) > 0:
            last_msg = state["messages"][-1]
            print(f"assistnat: {last_msg.content}")



if __name__ == "__main__":
    run_chatbot()














