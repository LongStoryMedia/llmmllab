import asyncio
from models import ModelProfile, NodeMetadata, Message, MessageRole
from composer.agents.chat_agent import ChatAgent
from composer.agents.base_agent import BaseAgent
from runner import PipelineFactory
from langchain.agents.middleware.todo import TodoListMiddleware
from models import PipelinePriority

# NOTE: This is a lightweight debug harness; in real E2E we use full workflow builder.

async def main():
    profile = ModelProfile(model_name="gpt-4o-mini", system_prompt="You are a helpful assistant.")
    node_meta = NodeMetadata(node_id="debug-node", node_name="debug_todos", node_type="test", user_id="debug-user", conversation_id=123)
    pipeline_factory = PipelineFactory()
    agent = ChatAgent(pipeline_factory=pipeline_factory, profile=profile, node_metadata=node_meta)
    agent.middleware = [TodoListMiddleware()]

    messages = [Message(role=MessageRole.USER, content=[{"text": "Plan a 5-step process to refactor legacy module"}])]

    response = await agent.run(messages=messages, priority=PipelinePriority.MEDIUM)
    print("Todos captured:")
    if response.todos:
        for i, td in enumerate(response.todos, 1):
            print(i, td.title, td.status, td.priority)
    else:
        print("No todos returned.")

if __name__ == "__main__":
    asyncio.run(main())
