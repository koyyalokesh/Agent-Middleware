from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import (
    before_model,
    after_model,
    wrap_model_call,
    wrap_tool_call
)

load_dotenv()

@tool
def order_status(order_id:str)->str:
    """Get the status of an order using its id."""
    return f"{order_id.upper()} : packed, ships tomorrow"


#Befor model , runs before the model is called

@before_model
def show_before(state, runtime):
    #print("state:", state)
    print(
        f"[before model]"
        f"{len(state['messages'])} message(s) going to the model"
    )
    return None  # None means, Im not changing the state


#Wrap model call, around the model call
@wrap_model_call
def time_the_model(request, handler):
    #print("request:", request)
    #we can inspect the request before sending it to the model
    
    print(
        f"[wrap_model_call]"
        f"tools offered: {[ x.name for x in request.tools]}"
    )
    response = handler(request)
    print("[wrap_model_call] model has replied")
    
    #Return the model response so the agent can continue.
    return response


#After Model, runs immediatley after the model responds

@after_model
def show_after(state, runtime):
    #get the latest message produced by the model
    last = state['messages'][-1]
    
    #check whether the model requested any tools.
    asked = [
        call["name"]
        for call in getattr(last, "tool_calls", []) or []
    ]
    
    print(
        f"[after model] asked for :"
        f"{asked or 'nothing, this is the final answer'}"
    )
    
    #we are only observing here
    return None

#wrap tool call
#this middleware sits around every tool execution

@wrap_tool_call
def show_tool(request, handler):
    
    #we can see which tool the agent is about to execute
    #and what arguments are being passed to it
    
    print(
        f"[wrap_tool_call]"
        f"running {request.tool_call['name']} "
        f"with {request.tool_call['args']}"
    )
    
    #continue the execution and actually run the tool.
    result = handler(request)
    
    #we are back after the tool has completed.
    print("[wrap_tool_call] tool finished")
    
    #Return the tool result so the agent can continue
    return result


agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[order_status],
    system_prompt="you are a support agent.",
    middleware=[
        show_before,
        time_the_model,
        show_after,
        show_tool
    ]
)

result = agent.invoke({
    "messages":[{
        "role":"user", "content":"where is my order ORD-1002?"
    }]
})

print()
print("Answer:", result["messages"][-1].content)