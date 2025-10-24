import os
import sys
import sqlite3
from pathlib import Path
import json
import logging
import threading
import time
from datetime import datetime, timedelta, UTC
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import re

# Load .env file automatically if present
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
except ImportError:
    # Fallback for systems without python-dotenv
    pass

# Correct imports for modern LangChain (v1.0)
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import StructuredTool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain.agents import create_agent
from langchain_community.chat_message_histories import ChatMessageHistory
from pydantic import BaseModel, Field

# Hugging Face and Vector Store dependencies
from huggingface_hub import InferenceClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Rich terminal output
try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.prompt import Prompt
except ImportError:
    # Dummy classes if rich is not available
    class Console:
        def print(self, *args, **kwargs): print(*args)
    class Markdown:
        def __init__(self, text): self.text = text
        def __str__(self): return self.text
    class Panel:
        def __init__(self, text, **kwargs): self.text = text
        def __str__(self): return str(self.text)
    class Prompt:
        @staticmethod
        def ask(prompt): return input(prompt)

# Windows notifications and audio (optional)
try:
    import winsound
    from win10toast import ToastNotifier
    WINDOWS_NOTIFICATIONS_AVAILABLE = True
except ImportError:
    WINDOWS_NOTIFICATIONS_AVAILABLE = False


# Small HF chat wrapper to normalize the InferenceClient for use with LangChain.
class HFChatLLM(BaseChatModel):
    client: "InferenceClient"
    model: str

    def __init__(self, client: "InferenceClient", model: str):
        super().__init__(client=client, model=model)

    def bind_tools(self, tools=None, **kwargs):
        return self

    @property
    def _llm_type(self) -> str:
        return "hf-chat"

    @property
    def _identifying_params(self) -> dict:
        return {"model": self.model}

    def invoke(self, prompt: List[Any], **kwargs) -> AIMessage:
        messages = []
        for p in prompt:
            role = "user"
            if isinstance(p, AIMessage): role = "assistant"
            elif isinstance(p, HumanMessage): role = "user"
            else: role = getattr(p, 'type', 'user')
            messages.append({"role": role, "content": str(p.content)})
        try:
            response = self.client.chat_completion(
                messages=messages, model=self.model,
                max_tokens=kwargs.get('max_tokens', 1024),
                temperature=kwargs.get('temperature', 0.7)
            )
            content = response.choices[0].message.content
            return AIMessage(content=content or "")
        except Exception as e:
            logger.error(f"Error during HF API call: {e}")
            return AIMessage(content="Sorry, I encountered an error while processing your request.")

    async def ainvoke(self, prompt: List[Any], **kwargs) -> AIMessage:
        # For simplicity, just call invoke
        return self.invoke(prompt, **kwargs)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        response = self.invoke(messages, **kwargs)
        generation = ChatGeneration(message=response)
        return ChatResult(generations=[generation])

# Configure logging
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('agent.log'), logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Constants
DB_PATH = "assistant_memory.db"
VECTOR_STORE_PATH = "memory_vectors"
DEFAULT_MODEL = os.environ.get("MODEL") or "mistralai/Mistral-7B-Instruct-v0.2"


# ============================================================================
# Data Models
# ============================================================================
@dataclass
class Memory:
    id: int
    content: str
    tags: List[str]
    created_at: str
    importance: int = 5
    context: Optional[str] = None
    summary: Optional[str] = None

    @classmethod
    def from_row(cls, row: tuple) -> 'Memory':
        return cls(id=row[0], content=row[1], tags=json.loads(row[2] or '[]'), created_at=row[3],
                   importance=row[4], context=row[5], summary=row[6])

@dataclass
class Reminder:
    id: int
    memory_id: int
    remind_at: str
    done: bool
    created_at: str
    recurrence: Optional[str] = None
    snoozed_until: Optional[str] = None

    @classmethod
    def from_row(cls, row: tuple) -> 'Reminder':
        return cls(id=row[0], memory_id=row[1], remind_at=row[2], done=bool(row[3]), created_at=row[4],
                   recurrence=row[5], snoozed_until=row[6])

# ============================================================================
# Pydantic Models for Tool Inputs
# ============================================================================
class SaveMemoryInput(BaseModel):
    content: str = Field(description="The content to remember")
    tags: Optional[List[str]] = Field(default=None, description="Tags for categorization")
    importance: Optional[int] = Field(default=5, description="Importance level 1-10")

class FindMemoriesInput(BaseModel):
    query: str = Field(description="Search query for finding memories")
    limit: Optional[int] = Field(default=5, description="Maximum number of results")

class SetReminderInput(BaseModel):
    memory_content: str = Field(description="What to be reminded about")
    when: str = Field(description="When to remind (e.g., 'in 30 minutes', 'tomorrow at 9am', 'in 2 hours')")

class ShowRemindersInput(BaseModel):
    show_completed: Optional[bool] = Field(default=True, description="Whether to include completed reminders in the list")


# ============================================================================
# Enhanced Memory Store
# ============================================================================
class EnhancedMemoryStore:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.embeddings: Optional[HuggingFaceEmbeddings] = None
        self.vector_store: Optional[FAISS] = None
        self._ensure_db()
        self._init_embeddings()

    def _ensure_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute("CREATE TABLE IF NOT EXISTS memories (id INTEGER PRIMARY KEY, content TEXT NOT NULL, tags TEXT, created_at TEXT NOT NULL, importance INTEGER DEFAULT 5, context TEXT, summary TEXT)")
            cur.execute("CREATE TABLE IF NOT EXISTS reminders (id INTEGER PRIMARY KEY, memory_id INTEGER, remind_at TEXT NOT NULL, done INTEGER DEFAULT 0, created_at TEXT NOT NULL, recurrence TEXT, snoozed_until TEXT, FOREIGN KEY(memory_id) REFERENCES memories(id))")
            conn.commit()

    def _init_embeddings(self):
        try:
            self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            self._load_or_rebuild_vector_store()
        except Exception as e:
            logger.warning(f"Could not initialize embeddings or vector store: {e}")

    def _load_or_rebuild_vector_store(self):
        vector_path = Path(VECTOR_STORE_PATH)
        if vector_path.exists() and self.embeddings:
            try:
                self.vector_store = FAISS.load_local(str(vector_path), self.embeddings, allow_dangerous_deserialization=True)
                logger.info("Vector store loaded from disk.")
                return
            except Exception as e:
                logger.warning(f"Failed to load vector store from disk, rebuilding: {e}")

        memories = self.get_all_memories()
        if not memories or not self.embeddings:
            logger.info("No memories to build vector store from, or embeddings not initialized.")
            return

        texts = [m.content for m in memories]
        metadatas = [{"id": m.id} for m in memories]
        self.vector_store = FAISS.from_texts(texts, self.embeddings, metadatas=metadatas)
        self.vector_store.save_local(str(vector_path))
        logger.info(f"Vector store rebuilt with {len(memories)} entries.")

    def save_memory(self, content: str, tags: Optional[List[str]] = None, importance: int = 5, context: Optional[str] = None) -> Memory:
        with sqlite3.connect(self.db_path) as conn:
            now = datetime.now(UTC).isoformat()
            cur = conn.cursor()
            cur.execute("INSERT INTO memories (content, tags, created_at, importance, context) VALUES (?, ?, ?, ?, ?)",
                        (content, json.dumps(tags or []), now, importance, context))
            conn.commit()
            memory_id = cur.lastrowid
            memory = Memory(id=memory_id, content=content, tags=(tags or []), created_at=now, importance=importance, context=context)

        if self.vector_store:
            self.vector_store.add_texts([content], metadatas=[{"id": memory.id}])
            self.vector_store.save_local(VECTOR_STORE_PATH)

        return memory

    def semantic_search(self, query: str, limit: int = 5) -> List[Memory]:
        if not self.vector_store:
            return []
        results = self.vector_store.similarity_search(query, k=limit)
        if not results:
            return []
        ids = [r.metadata["id"] for r in results]
        with sqlite3.connect(self.db_path) as conn:
            placeholders = ','.join('?' * len(ids))
            rows = conn.execute(f"SELECT * FROM memories WHERE id IN ({placeholders})", ids).fetchall()
            return [Memory.from_row(r) for r in rows]

    def set_reminder(self, memory_id: int, remind_at: datetime, recurrence: Optional[str] = None):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("INSERT INTO reminders (memory_id, remind_at, created_at, recurrence) VALUES (?, ?, ?, ?)",
                         (memory_id, remind_at.isoformat(), datetime.now(UTC).isoformat(), recurrence))
            conn.commit()

    def get_due_reminders(self) -> List[Tuple[Reminder, Memory]]:
        with sqlite3.connect(self.db_path) as conn:
            now = datetime.now(UTC).isoformat()
            rows = conn.execute("SELECT r.*, m.* FROM reminders r JOIN memories m ON r.memory_id = m.id WHERE r.done = 0 AND r.remind_at <= ?", (now,)).fetchall()
            return [(Reminder.from_row(row[:7]), Memory.from_row(row[7:])) for row in rows]

    def mark_reminder_done(self, reminder_id: int):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE reminders SET done = 1 WHERE id = ?", (reminder_id,))
            conn.commit()

    def get_all_reminders(self, include_completed: bool = False) -> List[Tuple[Reminder, Memory]]:
        with sqlite3.connect(self.db_path) as conn:
            query = "SELECT r.*, m.* FROM reminders r JOIN memories m ON r.memory_id = m.id"
            if not include_completed:
                query += " WHERE r.done = 0"
            query += " ORDER BY r.remind_at ASC"
            rows = conn.execute(query).fetchall()
            return [(Reminder.from_row(row[:7]), Memory.from_row(row[7:])) for row in rows]

    def get_all_memories(self) -> List[Memory]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT * FROM memories").fetchall()
            return [Memory.from_row(r) for r in rows]


# ============================================================================
# Advanced Chat Agent
# ============================================================================
class AdvancedChatAgent:
    def __init__(self, model_id: str = DEFAULT_MODEL):
        self.store = EnhancedMemoryStore()
        self.console = Console()
        self.model_id = model_id
        self.llm = self._init_llm(model_id=self.model_id)
        self.tools = self._create_tools()
        self.memory = ChatMessageHistory()
        self.agent_executor = self._create_agent_executor()
        self._start_reminder_daemon()
        logger.info(f"Agent initialized with model: {self.model_id}")

    def _init_llm(self, model_id: str):
        api_key = os.environ.get("HF_API_KEY") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        if not api_key:
            logger.error("HF_API_KEY not found in environment. Please set it in a .env file.")
            sys.exit(1)
        return HFChatLLM(client=InferenceClient(token=api_key), model=model_id)

    def _create_tools(self) -> List:
        def save_memory_tool(content: str, tags: Optional[List[str]] = None, importance: int = 5) -> str:
            memory = self.store.save_memory(content, tags=tags, importance=importance)
            return f"✓ Saved memory #{memory.id}: '{content[:50]}...'"
        def find_memories_tool(query: str, limit: int = 5) -> str:
            memories = self.store.semantic_search(query, limit)
            if not memories: return "No relevant memories found."
            return f"Found {len(memories)} memories:\n" + "\n".join(f"- #{m.id}: {m.content[:100]}" for m in memories)
        def set_reminder_tool(memory_content: str, when: str) -> str:
            try:
                minutes = self._parse_time_expression(when)
                remind_at = datetime.now(UTC) + timedelta(minutes=minutes)
                memory = self.store.save_memory(memory_content, tags=["reminder"])
                self.store.set_reminder(memory.id, remind_at)
                return f"✓ Reminder set for {remind_at.strftime('%Y-%m-%d %H:%M')} UTC"
            except Exception as e: return f"Failed to set reminder: {e}"
        def show_reminders_tool(show_completed: bool = True) -> str:
            reminders = self.store.get_all_reminders(include_completed=show_completed)
            if not reminders:
                return "No reminders found." if not show_completed else "No reminders found (including completed)."
            
            now = datetime.now(UTC)
            result = []
            for reminder, memory in reminders:
                remind_time = datetime.fromisoformat(reminder.remind_at.replace('Z', '+00:00'))
                status = "✅ COMPLETED" if reminder.done else ("⏰ DUE NOW" if remind_time <= now else "⏳ PENDING")
                time_str = remind_time.strftime('%Y-%m-%d %H:%M UTC')
                result.append(f"• {status}: {memory.content}\n  Scheduled: {time_str}")
            
            header = "Your Reminders:" if not show_completed else "All Reminders (including completed):"
            return f"{header}\n\n" + "\n\n".join(result)
        return [
            StructuredTool.from_function(func=save_memory_tool, name="save_memory", description="Save important information or facts.", args_schema=SaveMemoryInput),
            StructuredTool.from_function(func=find_memories_tool, name="find_memories", description="Search saved memories.", args_schema=FindMemoriesInput),
            StructuredTool.from_function(func=set_reminder_tool, name="set_reminder", description="Set a reminder for the future.", args_schema=SetReminderInput),
            StructuredTool.from_function(func=show_reminders_tool, name="show_reminders", description="REQUIRED: Use this tool IMMEDIATELY when users ask about their reminders, schedule, calendar, or appointments. Shows all reminders with status indicators.", args_schema=ShowRemindersInput),
        ]

    def _create_agent_executor(self):
        # This is the full, unabridged system prompt
        system_message = """You are an advanced AI assistant with a perfect memory and proactive intelligence.

Your capabilities:
- You have a long-term memory to store and recall information.
- You can set reminders that will trigger at a specific time.
- You can show users their complete calendar of all reminders (past and future).
- You must decide when to use these tools based on the conversation.

Guidelines:
1.  **Be Proactive:** AUTOMATICALLY save memories when the user shares important information (names, projects, deadlines, preferences). Do not ask for permission.
2.  **Use Your Memory:** PROACTIVELY search your memory when the user asks a question that might relate to past conversations.
3.  **Offer Assistance:** OFFER to set reminders when the user mentions future tasks or events.
4.  **Show Calendar:** CRITICAL: When users ask about their reminders, schedule, calendar, appointments, or use phrases like "check my reminders", "what's on my calendar", "show reminders", "my schedule", "calendar", "appointments" - you MUST use the show_reminders tool immediately. Do not respond with generic text - use the tool first.
5.  **Be Conversational:** Interact naturally. Do not announce your actions like "I am saving this to memory." The tool's output is your confirmation.
6.  **Rate Importance:** Use the importance score (1-10) when saving memories. Critical info is 9-10, important details are 7-8, and casual facts are 4-6."""

        agent = create_agent(model=self.llm, tools=self.tools, system_prompt=system_message)
        return agent

    def _parse_time_expression(self, expr: str) -> int:
        expr = expr.lower().strip()
        total_minutes = 0
        patterns = {
            r'(\d+)\s*(min|minute|minutes)': lambda m: int(m.group(1)),
            r'(\d+)\s*(hr|hour|hours)': lambda m: int(m.group(1)) * 60,
            r'(\d+)\s*(day|days)': lambda m: int(m.group(1)) * 1440,
        }
        for pattern, converter in patterns.items():
            for match in re.finditer(pattern, expr):
                total_minutes += converter(match)
        if 'tomorrow' in expr:
            total_minutes += 1440
        return total_minutes if total_minutes > 0 else 60 # Default to 1 hour

    def _start_reminder_daemon(self):
        def check():
            toaster = ToastNotifier() if WINDOWS_NOTIFICATIONS_AVAILABLE else None
            while True:
                try:
                    for reminder, memory in self.store.get_due_reminders():
                        # Console notification
                        self.console.print(Panel(f"[bold yellow]⏰ REMINDER[/bold yellow]\n\n{memory.content}", title="Reminder", border_style="yellow"))

                        # Windows desktop notification
                        if WINDOWS_NOTIFICATIONS_AVAILABLE and toaster:
                            try:
                                toaster.show_toast(
                                    "AI Assistant Reminder",
                                    memory.content,
                                    duration=10,
                                    threaded=True
                                )
                            except Exception as e:
                                logger.warning(f"Failed to show desktop notification: {e}")

                        # Windows audio alert
                        if WINDOWS_NOTIFICATIONS_AVAILABLE:
                            try:
                                # Play system asterisk sound (you can change this to different system sounds)
                                winsound.MessageBeep(winsound.SND_ALIAS)
                            except Exception as e:
                                logger.warning(f"Failed to play audio alert: {e}")

                        self.store.mark_reminder_done(reminder.id)
                except Exception as e:
                    logger.error(f"Reminder check error: {e}")
                time.sleep(30)
        threading.Thread(target=check, daemon=True).start()

    def chat(self):
        self.console.print(Panel(f"[bold cyan]Advanced AI Memory Assistant[/bold cyan]\nModel: [white]{self.model_id}[/white]", title="Welcome", border_style="cyan"))
        while True:
            try:
                user_input = Prompt.ask("\n[bold green]You[/bold green]")
                if user_input.lower() in ("quit", "exit", "bye"):
                    break
                self.memory.add_user_message(user_input)
                result = self.agent_executor.invoke({"messages": self.memory.messages})
                output = result["messages"][-1].content
                self.memory.add_ai_message(output)
                self.console.print(Markdown(f"\n**Assistant**: {output}"))
            except (KeyboardInterrupt, EOFError):
                break
            except Exception as e:
                logger.error(f"Chat error: {e}", exc_info=True)
                self.console.print(f"[bold red]An unexpected error occurred: {e}[/bold red]")
        self.console.print("\n[yellow]Goodbye! Your memories are saved.[/yellow]")

def main():
    try:
        model_to_use = os.environ.get("MODEL", DEFAULT_MODEL)
        agent = AdvancedChatAgent(model_id=model_to_use)
        agent.chat()
    except Exception as e:
        logger.critical(f"A fatal error occurred during initialization: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()