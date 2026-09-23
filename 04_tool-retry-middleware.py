from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.agents.middleware import ToolRetryMiddleware
from langchain_core.tools import tool

load_dotenv()

attempts = {"count": 0}


@tool
def order_status(order_id: str) -> str:
    """Get the status of an order using its id."""

    attempts["count"] += 1

    print(f"Attempt {attempts['count']} to reach the order service")

    # Deliberately fail the first two attempts
    if attempts["count"] < 3:
        raise RuntimeError("order service unavailable")

    return f"{order_id.upper()}: packed, ships tomorrow"


agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[order_status],
    system_prompt="You are a support agent. Look up the order before answering.",
    middleware=[
        ToolRetryMiddleware(
            max_retries=5,
            initial_delay=0.2,
            backoff_factor=1.5,
        )
    ],
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Where is my order ORD-1002?"
            }
        ]
    }
)

print(f"Answer: {result['messages'][-1].content}")
print(f"Total Attempts: {attempts['count']}")