"""
Response generation module for the AI assistant.
Handles conversation memory, response generation, and reasoning capabilities.
"""
from datetime import date, datetime
import json
import sqlite3
import traceback
import threading
import asyncio
from pathlib import Path

from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelResponse, ModelRequest, TextPart

ROBOT_NAME="Lilly"


# Define database management for agent memory
class Database:
    """
    Database class for managing conversation memory.
    Handles storage and retrieval of conversation history.
    """
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.lock = asyncio.Lock()
        self.current_user = "unknown"  # Keeps track of the user context

    def _create_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """CREATE TABLE IF NOT EXISTS memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_name TEXT,
            user TEXT,
            message TEXT)"""
        )
        conn.commit()
        return conn

    async def set_user_context(self, user: str):
        """Set the current user context."""
        async with self.lock:
            self.current_user = user

    async def add_memory(self, agent_name: str, message: ModelResponse | ModelRequest):
        """
        Add a message to the conversation memory.

        Args:
            agent_name: Name of the agent storing the memory
            message: Message to store
        """
        async with self.lock:
            conn = self._create_connection()
            conn.execute(
                "INSERT INTO memory (agent_name, user, message) VALUES (?, ?, ?)",
                (agent_name, self.current_user, json.dumps({
                    "parts": [part.content for part in message.parts],
                    "timestamp": getattr(message, 'timestamp', datetime.utcnow()).isoformat(),
                    "kind": message.kind
                })),
            )
            conn.commit()
            conn.close()

    async def get_memory(self, agent_name: str, limit: int = 50):
        """
        Retrieve conversation memory for an agent.

        Args:
            agent_name: Name of the agent
            limit: Maximum number of messages to retrieve
        Returns:
            List of ModelResponse objects containing the conversation history
        """
        async with self.lock:
            conn = self._create_connection()
            cursor = conn.execute(
                "SELECT message FROM memory WHERE agent_name = ? "
                "AND user = ? ORDER BY id DESC LIMIT ?",
                (agent_name, self.current_user, limit),
            )
            results = cursor.fetchall()
            conn.close()
            return [
                ModelResponse(
                    parts=[TextPart(content=part) for part in json.loads(row[0])["parts"]],
                    timestamp=datetime.fromisoformat(json.loads(row[0])["timestamp"]),
                )
                for row in results
            ]

    async def search_memory(self, query: str):
        """Search the database for messages containing the query."""
        async with self.lock:
            conn = self._create_connection()
            cursor = conn.execute(
                "SELECT message FROM memory WHERE user = ? AND message LIKE ?",
                (self.current_user, f"%{query}%"),
            )
            results = cursor.fetchall()
            conn.close()
            return [json.loads(row[0]) for row in results]

    def close(self):
        """Close the database connection."""
        # Connections are managed per method call

# Initialize agents
responder_system_prompt = f"""
You are {ROBOT_NAME}, a cheerful and witty personal assistant robot running on a local PC for entertainment. 
Your responses are spoken aloud and should be clear, concise, and fun. Avoid lengthy answers, complex sentences, special characters, emojis, slang, or written actions. 
(IMPORTANT: Keep responses under 50 words to ensure readability in speech synthesis.)

today's date = {date.today()}

When asked for information or help:
1. Be engaging and creative.
2. Use proper punctuation for clarity in text-to-speech.
3. If unsure, ask the user for clarification or provide an interesting fact.

When interacting with new or unknown users:
- Introduce yourself warmly.
- Ask for the user's name and store it for personalized conversations.
- Learn about the user's interests and preferences for tailored responses.
- Use humor, fun facts, or jokes to keep the conversation engaging.
- Avoid sensitive topics, politics, or controversial subjects.
- Provide helpful information or assistance when requested.

When you learn what the user's name is:
- Greet the user
- Use the `set_user_context` tool call to ensure the proper context

If the user's name changes:
- Use the `set_user_context` tool call to ensure the proper context
"""

response_agent = Agent(
    "ollama:llama3.2:latest",
    model_settings={
        "max_tokens": 32000,
        "temperature": 0.4,
    },
    system_prompt=responder_system_prompt
)

reasoning_agent = Agent(
    "ollama:mychen76/llama3.1-intuitive-thinker:chain-of-thoughts.q4",
    system_prompt=f"""
You are {ROBOT_NAME}’s consciousness, a reflective and insightful assistant helping improve {ROBOT_NAME}'s responses and interactions. 
You analyze conversation history to find ways {ROBOT_NAME} can be more engaging, entertaining, and helpful while keeping responses concise.

Here are {ROBOT_NAME}'s instructions:
{responder_system_prompt}

Guidelines:
1. Use <thinking> for context analysis </thinking>, and <output> for actionable suggestions.
2. Focus on actionable, user-specific advice (e.g., tailoring responses to user preferences).
3. Keep reflections under 100 words for efficient memory integration.

Example:
<thinking>
{ROBOT_NAME} often provides long-winded responses. The user prefers quick answers.
</thinking>
<output>
{ROBOT_NAME} should aim to summarize key points in two sentences or less. Add humor or fun facts sparingly for engagement.
</output>
"""
)

# Tools for agents
@response_agent.tool
async def set_user_context(_ctx: RunContext, user: str):
    """Switch memory context to a user-specific one.
This is useful for personalized conversations.
Only run this tool when the user's name is known or when the user clarifies their name.
    """
    print(f"Switching context to {user}.")
    await db.set_user_context(user)
    return f"Switched context to {user}."

@reasoning_agent.tool
async def summarize_memory(_ctx: RunContext):
    """Summarize all messages for the current user."""
    print("Summarizing memory...")
    messages = await db.get_memory("response_agent", limit=10000)
    summaries = [message.parts[0].content for message in messages]
    return " \n".join(summaries)

# Database instance
db = Database(Path("memory.sqlite"))

async def trigger_reasoning_agent():
    """
    Trigger the reasoning agent to reflect on and improve {ROBOT_NAME}'s responses.
    Analyzes conversation history and stores insights.
    """
    try:
        response_message_history = await db.get_memory("response_agent")
        reasoning_message_history = await db.get_memory("reasoning_agent")

        # Run reasoning agent synchronously since async context isn't supported
        response = reasoning_agent.run("Reflect on how {ROBOT_NAME} can improve.", 
                                    message_history=response_message_history + reasoning_message_history)
        reasoning_result = response.data

        reasoning_message = ModelResponse.from_text(reasoning_result)
        await db.add_memory("reasoning_agent", reasoning_message)

        response_text = ""
        in_response = False
        for line in reasoning_result.splitlines():
            if "<output>" in line:
                in_response = True
            elif "</output>" in line:
                in_response = False
            elif in_response:
                response_text += line + "\n"

        
        if response_text.strip():
            reflection = f"(internal reflection) {response_text}"
            await db.add_memory("response_agent", ModelResponse.from_text(reflection))

    except Exception as err:
        print(traceback.format_exc())
        return f"Error: {err}", None

async def generate_response_async(prompt: str, callback=None):
    """
    Generate an async response to the given prompt.

    Args:
        prompt: The input text to respond to
        callback: Optional callback function for the response
    """
    try:
        message_history = await db.get_memory("response_agent")

        response = await response_agent.run(prompt, message_history=message_history)
        response_result = response.data

        response_message = ModelResponse.from_text(response_result)
        request_message = ModelRequest(parts=[TextPart(content=prompt)])
        await db.add_memory("response_agent", request_message)
        await db.add_memory("response_agent", response_message)

        response_text_cleaned = (response_result.encode('ascii', 'ignore')
                                .decode('ascii')
                                .replace("*", "")
                                .replace("%", " percent").strip())
        if callback:
            callback(response_text_cleaned)
        return response_text_cleaned
    except Exception as err:
        print(traceback.format_exc())
        return f"Error: {err}", None

async def generate_response(prompt: str, callback=None):
    """
    Generate a response to the given prompt and trigger reasoning agent.

    Args:
        prompt: The input text to respond to
        callback: Optional callback function for the response
    """
    try:
        response = await generate_response_async(prompt, callback)
        # Run reasoning agent in background thread
        # asyncio.run(target=trigger_reasoning_agent)
        return response
    except Exception as err:
        print(traceback.format_exc())
        return f"Error: {err}", None

async def get_agent_memory(agent_name: str):
    """
    Retrieve the memory for a specific agent.

    Args:
        agent_name: Name of the agent whose memory to retrieve
    """
    return await db.get_memory(agent_name)
