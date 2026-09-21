from dotenv import load_dotenv
from dataclasses import dataclass
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain.agents.middleware import dynamic_prompt


load_dotenv()

@dataclass
class Customer:
    name:str
    plan:str
    language:str

@tool
def order_status(order_id:str)->str:
    """get the status of an order using its id."""
    return f"{order_id.upper()}: Packed, ships tomorrow."

@dynamic_prompt
def support_prompt(request):
    customer = request.runtime.context
    lines = [
        f"You are a support agent for an online store. The customer is {customer.name}.",
        "Look up the order before answering, never guess.",
        "Write like a chat reply, no email signature.",
    ]
    
    if customer.plan == "premium":
        lines.append("This is a premium customer, apologise for any delay and offer a callback.")
    else:
        lines.append("This is a free plan customer, keep the answer to two lines.")
    
    lines.append(f"Reply in {customer.language}.")
    prompt = " ".join(lines)
    print("[prompt used]", prompt)
    print()
    
    return prompt

agent = create_agent(
    model ="gpt-4o-mini",
    tools=[order_status],
    middleware=[support_prompt],
    context_schema=Customer,
)

question = {
    "messages":[
        {
            "role":"user",
            "content":"Where is my order ORD-1002?"
        }
    ]
}

userObjects = [
     Customer(name="Lokesh",plan="premium",language="English"),
     Customer(name="Dhoni",plan="free",language="English"),
]

for customer in userObjects:
    result = agent.invoke(question, context=customer)
    print(f"To {customer.name} ({customer.plan}):")
    print(result["messages"][-1].content)