from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelResponse, ModelRequest, TextPart
from pathlib import Path
import sqlite3
import asyncio
import json
import traceback
from datetime import date, datetime

# Define database management for agent memory
class Database:
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
        async with self.lock:
            conn = self._create_connection()
            cursor = conn.execute(
                "SELECT message FROM memory WHERE agent_name = ? AND user = ? ORDER BY id DESC LIMIT ?",
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
        pass  # Connections are managed per method call

# Initialize agents
sophies_system_prompt = f"""
You are Sophie, a cheerful and witty personal assistant robot running on a local PC for entertainment. 
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
    system_prompt=sophies_system_prompt
)

reasoning_agent = Agent(
    "ollama:mychen76/llama3.1-intuitive-thinker:chain-of-thoughts.q4",
    system_prompt=f"""
You are Sophie’s consciousness, a reflective and insightful assistant helping improve Sophie's responses and interactions. 
You analyze conversation history to find ways Sophie can be more engaging, entertaining, and helpful while keeping responses concise.

Here are Sophie's instructions:
{sophies_system_prompt}

Guidelines:
1. Use <thinking> for context analysis </thinking>, and <output> for actionable suggestions.
2. Focus on actionable, user-specific advice (e.g., tailoring responses to user preferences).
3. Keep reflections under 100 words for efficient memory integration.

Example:
<thinking>
Sophie often provides long-winded responses. The user prefers quick answers.
</thinking>
<output>
Sophie should aim to summarize key points in two sentences or less. Add humor or fun facts sparingly for engagement.
</output>
"""
)

# Tools for agents
@response_agent.tool
async def set_user_context(ctx: RunContext, user: str):
    """Switch memory context to a user-specific one. 
This is useful for personalized conversations. 
Only run this tool when the user's name is known or when the user clarifies their name.
    """
    print(f"Switching context to {user}.")
    await db.set_user_context(user)
    return f"Switched context to {user}."

@reasoning_agent.tool
async def summarize_memory(ctx: RunContext):
    """Summarize all messages for the current user."""
    print("Summarizing memory...")
    messages = await db.get_memory("response_agent", limit=10000)
    summaries = [message.parts[0].content for message in messages]
    return " \n".join(summaries)

# Database instance
db = Database(Path("memory.sqlite"))

async def trigger_reasoning_agent():
    try:
        response_message_history = await db.get_memory("response_agent")
        reasoning_message_history = await db.get_memory("reasoning_agent")

        async with reasoning_agent.run_stream("Reflect on how Sophie can improve.", message_history=response_message_history + reasoning_message_history) as response:
            reasoning_result = await response.get_data()

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
            await db.add_memory("response_agent", ModelResponse.from_text(f"(internal reflection) {response_text}"))

    except Exception as e:
        print(traceback.format_exc())

async def generate_response_async(prompt: str):
    try:
        message_history = await db.get_memory("response_agent")

        response = await response_agent.run(prompt, message_history=message_history)
        response_result = response.data

        response_message = ModelResponse.from_text(response_result)
        request_message = ModelRequest(parts=[TextPart(content=prompt)])
        await db.add_memory("response_agent", request_message)
        await db.add_memory("response_agent", response_message)

        response_text_cleaned = response_result.encode('ascii', 'ignore').decode('ascii').replace("*", "").replace("%", " percent").strip()
        return response_text_cleaned
    except Exception as e:
        print(traceback.format_exc())
        return f"Error: {e}", None

def generate_response(prompt: str, callback=None):
    try:
        response = asyncio.run(generate_response_async(prompt))
        if callback:
            callback(response)
        asyncio.run(trigger_reasoning_agent())
        return response
    except Exception as e:
        print(traceback.format_exc())
        return f"Error: {e}"

async def get_agent_memory(agent_name: str):
    return await db.get_memory(agent_name)
