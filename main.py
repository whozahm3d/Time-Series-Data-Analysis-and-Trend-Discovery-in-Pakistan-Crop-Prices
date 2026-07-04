"""
Pakistan Agricultural Price Intelligence System
Multi-agent entry point using Google ADK

Usage:
    # Interactive chat via ADK web UI (recommended)
    adk web agents/orchestrator

    # Interactive chat via ADK web UI
    adk web

    # Or run the orchestrator programmatically
    python main.py
"""

import asyncio
import sys

# Ensure UTF-8 output on Windows terminals (crop names may contain Unicode)
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# pyrefly: ignore [missing-import]
from google.adk.runners import Runner
# pyrefly: ignore [missing-import]
from google.adk.sessions import InMemorySessionService
# pyrefly: ignore [missing-import]
from google.genai.types import Content, Part

# Import the root orchestrator agent
from agents.orchestrator.agent import root_agent


async def run_query(query: str, session_id: str = "default") -> str:
    """Run a single query through the multi-agent pipeline."""
    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name="agri_price_intelligence",
        user_id="farmer_user",
        session_id=session_id,
    )

    runner = Runner(
        agent=root_agent,
        app_name="agri_price_intelligence",
        session_service=session_service,
    )

    message = Content(role="user", parts=[Part(text=query)])
    response_text = ""

    async for event in runner.run_async(
        user_id="farmer_user",
        session_id=session_id,
        new_message=message,
    ):
        if event.is_final_response() and event.content:
            for part in event.content.parts:
                if part.text:
                    response_text += part.text

    return response_text


async def interactive_session():
    """Run an interactive CLI session with the multi-agent system."""
    print("\n" + "=" * 70)
    print("  🌾 Pakistan Agricultural Price Intelligence System")
    print("  Multi-Agent AI powered by Google ADK")
    print("=" * 70)
    print("Type your question (or 'quit' to exit)\n")

    session_id = "cli_session_001"

    # Greet on startup
    greeting = await run_query("Hello, what can you help me with?", session_id)
    print(f"Agent: {greeting}\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_input.lower() in ("quit", "exit", "bye"):
            print("Agent: Goodbye! May your harvests be prosperous. 🌾")
            break

        if not user_input:
            continue

        print("\nAgent: (thinking...)")
        response = await run_query(user_input, session_id)
        print(f"Agent: {response}\n")


if __name__ == "__main__":
    # Quick smoke-test if called directly
    test_query = (
        "What is the average price of Apple (Ammre) in Lahore? "
        "Also forecast the next 3 months and give me a recommendation."
    )
    print(f"\nRunning test query: {test_query}\n")
    result = asyncio.run(run_query(test_query))
    print(result)
