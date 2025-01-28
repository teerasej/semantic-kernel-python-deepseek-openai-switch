import asyncio
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import ( 
    OpenAIChatCompletion,
    OpenAIChatPromptExecutionSettings
)
from semantic_kernel.contents.chat_history import ChatHistory

# Load environment variables from .env file
load_dotenv()

async def main():
    # Initialize the kernel
    kernel = Kernel()

    # Add OpenAI chat completion service
    api_key = os.getenv("API_KEY")  # Load API key from environment variable
    model_id = os.getenv("MODEL_ID")  # Load model ID from environment variable
    service_uri = os.getenv("SERVICE_URI")  # Load service URI from environment variable
    chat_completion = OpenAIChatCompletion(
        ai_model_id=model_id, 
        service_id="chat_completion",
        async_client=AsyncOpenAI(
            api_key=api_key,
            base_url=service_uri,
        )
    )
    kernel.add_service(chat_completion)

    # Create a chat history instance
    chat_history = ChatHistory()

    print("Type 'exit' to end the chat.")
    while True:
        # Get user input
        user_input = input("User: ")
        if user_input.lower() == "exit":
            break

        # Add user message to chat history
        chat_history.add_user_message(user_input)

        # Get AI response
        prompt = chat_history.to_prompt()
        ai_response = await kernel.invoke_prompt(prompt)

        # Display AI response
        print(f"Assistant: {ai_response}")

        # Add AI response to chat history
        # chat_history.add_assistant_message(ai_response)

if __name__ == "__main__":
    asyncio.run(main())