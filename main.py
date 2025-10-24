import os
import sys
import sqlite3
from pathlib import Path
import json
import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import re
from collections import defaultdict

# Load .env file automatically if present
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
except ImportError:
    pass

# LangChain imports
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration
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
    from rich.table import Table
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    class Console:
        def print(self, *args, **kwargs): print(*args)
    class Markdown:
        def __init__(self, text): self.text = text
    class Panel:
        def __init__(self, text, **kwargs): self.text = text
    class Prompt:
        @staticmethod
        def ask(prompt): return input(prompt)

# Windows notifications (optional)
try:
    import winsound
    # from win10toast import ToastNotifier  # Disabled due to Windows compatibility issues
    WINDOWS_NOTIFICATIONS_AVAILABLE = False  # Force disabled
except ImportError:
    WINDOWS_NOTIFICATIONS_AVAILABLE = False


# ============================================================================
# Custom LLM Wrapper
# ============================================================================
class HFChatLLM(BaseChatModel):
    """Hugging Face Chat Model wrapper with proper error handling and retry logic."""
    client: "InferenceClient"
    model: str
    max_retries: int = 3
    retry_delay: int = 2

    def __init__(self, client: "InferenceClient", model: str, max_retries: int = 3):
        super().__init__(client=client, model=model, max_retries=max_retries)

    def bind_tools(self, tools=None, **kwargs):
        return self

    @property
    def _llm_type(self) -> str:
        return "hf-chat"

    @property
    def _identifying_params(self) -> dict:
        return {"model": self.model}

    def _format_messages(self, prompt: List[Any]) -> List[Dict[str, str]]:
        """Convert various message formats to standard format."""
        messages = []
        for p in prompt:
            if isinstance(p, dict):
                role = p.get("role", "user")
                content = p.get("content", "")
            elif isinstance(p, SystemMessage):
                role = "system"
                content = str(p.content)
            elif isinstance(p, AIMessage):
                role = "assistant"
                content = str(p.content)
            elif isinstance(p, HumanMessage):
                role = "user"
                content = str(p.content)
            else:
                role = "user"
                content = str(p.content) if hasattr(p, 'content') else str(p)
            messages.append({"role": role, "content": content})
        return messages

    def invoke(self, prompt: List[Any], **kwargs) -> AIMessage:
        """Invoke the model with retry logic."""
        messages = self._format_messages(prompt)
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat_completion(
                    messages=messages,
                    model=self.model,
                    max_tokens=kwargs.get('max_tokens', 1024),
                    temperature=kwargs.get('temperature', 0.7)
                )
                content = response.choices[0].message.content
                return AIMessage(content=content or "")
            except Exception as e:
                logger.warning(f"API call attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    logger.error(f"All retry attempts failed: {e}")
                    return AIMessage(content="I apologize, but I'm having trouble connecting right now. Please try again in a moment.")

    async def ainvoke(self, prompt: List[Any], **kwargs) -> AIMessage:
        return self.invoke(prompt, **kwargs)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        response = self.invoke(messages, **kwargs)
        generation = ChatGeneration(message=response)
        return ChatResult(generations=[generation])


# ============================================================================
# Logging Configuration
# ============================================================================
def setup_logging():
    """Configure logging with rotation and proper formatting."""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Create logs directory if it doesn't exist
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    # File handler with rotation
    file_handler = logging.FileHandler(log_dir / 'assistant.log')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(log_format))
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING)  # Only warnings and errors to console
    console_handler.setFormatter(logging.Formatter(log_format))
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, console_handler]
    )

setup_logging()
logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================
DB_PATH = "assistant_memory.db"
VECTOR_STORE_PATH = "memory_vectors"
DEFAULT_MODEL = os.environ.get("MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
REMINDER_CHECK_INTERVAL = 30  # seconds


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
    last_accessed: Optional[str] = None

    @classmethod
    def from_row(cls, row: tuple) -> 'Memory':
        return cls(
            id=row[0],
            content=row[1],
            tags=json.loads(row[2] or '[]'),
            created_at=row[3],
            importance=row[4],
            context=row[5],
            summary=row[6],
            last_accessed=row[7] if len(row) > 7 else None
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Reminder:
    id: int
    memory_id: int
    remind_at: str
    done: bool
    created_at: str
    recurrence: Optional[str] = None
    snoozed_until: Optional[str] = None
    completed_at: Optional[str] = None

    @classmethod
    def from_row(cls, row: tuple) -> 'Reminder':
        return cls(
            id=row[0],
            memory_id=row[1],
            remind_at=row[2],
            done=bool(row[3]),
            created_at=row[4],
            recurrence=row[5],
            snoozed_until=row[6],
            completed_at=row[7] if len(row) > 7 else None
        )

    def to_dict(self) -> dict:
        return asdict(self)


# ============================================================================
# Pydantic Models for Tool Inputs
# ============================================================================
class SaveMemoryInput(BaseModel):
    content: str = Field(description="The content to remember")
    tags: Optional[List[str]] = Field(default=None, description="Tags for categorization")
    importance: Optional[int] = Field(default=5, ge=1, le=10, description="Importance level 1-10")
    context: Optional[str] = Field(default=None, description="Additional context")


class FindMemoriesInput(BaseModel):
    query: str = Field(description="Search query for finding memories")
    search_term: Optional[str] = Field(default=None, description="Alternative name for query - will be used if query not provided")
    limit: Optional[int] = Field(default=5, ge=1, le=20, description="Maximum number of results")


class FindRemindersInput(BaseModel):
    query: str = Field(description="Search query for finding reminders by content")
    search_term: Optional[str] = Field(default=None, description="Alternative name for query - will be used if query not provided")
    limit: Optional[int] = Field(default=5, ge=1, le=20, description="Maximum number of results")


class SetReminderInput(BaseModel):
    memory_content: Optional[str] = Field(default=None, description="What to be reminded about (the content/subject of the reminder)")
    when: Optional[str] = Field(default=None, description="When to remind (e.g., 'in 30 minutes', 'tomorrow at 9am', 'in 1 hour')")
    description: Optional[str] = Field(default=None, description="Alternative parameter for reminder content")
    time: Optional[str] = Field(default=None, description="Alternative parameter for when to remind")
    content: Optional[str] = Field(default=None, description="Alternative parameter for what to be reminded about")


class ShowRemindersInput(BaseModel):
    show_completed: Optional[bool] = Field(default=False, description="Whether to include completed reminders")


class UpdateMemoryInput(BaseModel):
    memory_id: int = Field(description="ID of the memory to update")
    content: Optional[str] = Field(default=None, description="New content")
    tags: Optional[List[str]] = Field(default=None, description="New tags")
    importance: Optional[int] = Field(default=None, ge=1, le=10, description="New importance level")


class DeleteMemoryInput(BaseModel):
    memory_id: int = Field(description="ID of the memory to delete")


class DeleteReminderInput(BaseModel):
    reminder_id: Optional[int] = Field(default=None, description="ID of specific reminder to delete")
    reminder_ids: Optional[List[int]] = Field(default=None, description="List of reminder IDs to delete multiple reminders at once")
    ids: Optional[List[int]] = Field(default=None, description="Alternative parameter for list of reminder IDs")
    id: Optional[int] = Field(default=None, description="Alternative parameter for single reminder ID")
    delete_all: Optional[bool] = Field(default=False, description="Set to True to delete all reminders")
    all: Optional[bool] = Field(default=False, description="Alternative parameter - set to True to delete all reminders")
    include_completed: Optional[bool] = Field(default=True, description="Include completed reminders when deleting all")


class UpdateReminderInput(BaseModel):
    reminder_id: int = Field(description="ID of the reminder to update")
    new_time: Optional[str] = Field(default=None, description="New time for the reminder (e.g., 'in 5 minutes', 'tomorrow at 3pm')")
    new_content: Optional[str] = Field(default=None, description="New content/description for the reminder")


# ============================================================================
# Enhanced Memory Store
# ============================================================================
class EnhancedMemoryStore:
    """Enhanced memory store with vector search, tagging, and better organization."""
    
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.embeddings: Optional[HuggingFaceEmbeddings] = None
        self.vector_store: Optional[FAISS] = None
        self._lock = threading.Lock()
        self._ensure_db()
        self._init_embeddings()

    def _ensure_db(self):
        """Create database tables with proper schema."""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            
            # Memories table with last_accessed tracking
            cur.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY,
                    content TEXT NOT NULL,
                    tags TEXT,
                    created_at TEXT NOT NULL,
                    importance INTEGER DEFAULT 5,
                    context TEXT,
                    summary TEXT,
                    last_accessed TEXT
                )
            """)
            
            # Migration: Add last_accessed column if it doesn't exist
            try:
                cur.execute("SELECT last_accessed FROM memories LIMIT 1")
            except sqlite3.OperationalError:
                logger.info("Adding last_accessed column to memories table")
                cur.execute("ALTER TABLE memories ADD COLUMN last_accessed TEXT")
            
            # Reminders table with completed_at tracking
            cur.execute("""
                CREATE TABLE IF NOT EXISTS reminders (
                    id INTEGER PRIMARY KEY,
                    memory_id INTEGER NOT NULL,
                    remind_at TEXT NOT NULL,
                    done INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    recurrence TEXT,
                    snoozed_until TEXT,
                    completed_at TEXT,
                    FOREIGN KEY(memory_id) REFERENCES memories(id) ON DELETE CASCADE
                )
            """)
            
            # Migration: Add completed_at column if it doesn't exist
            try:
                cur.execute("SELECT completed_at FROM reminders LIMIT 1")
            except sqlite3.OperationalError:
                logger.info("Adding completed_at column to reminders table")
                cur.execute("ALTER TABLE reminders ADD COLUMN completed_at TEXT")
            
            # Create indices for better performance
            cur.execute("CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_reminders_remind_at ON reminders(remind_at)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_reminders_done ON reminders(done)")
            
            conn.commit()
            logger.info("Database initialized successfully")

    def _init_embeddings(self):
        """Initialize embeddings and vector store."""
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            self._load_or_rebuild_vector_store()
            logger.info("Embeddings initialized successfully")
        except Exception as e:
            logger.warning(f"Could not initialize embeddings: {e}")

    def _load_or_rebuild_vector_store(self):
        """Load existing vector store or rebuild from database."""
        vector_path = Path(VECTOR_STORE_PATH)
        
        if vector_path.exists() and self.embeddings:
            try:
                self.vector_store = FAISS.load_local(
                    str(vector_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info("Vector store loaded from disk")
                return
            except Exception as e:
                logger.warning(f"Failed to load vector store, rebuilding: {e}")

        # Rebuild vector store
        memories = self.get_all_memories()
        if not memories or not self.embeddings:
            logger.info("No memories to build vector store from")
            return

        texts = [m.content for m in memories]
        metadatas = [{"id": m.id, "importance": m.importance, "tags": json.dumps(m.tags)} for m in memories]
        
        self.vector_store = FAISS.from_texts(texts, self.embeddings, metadatas=metadatas)
        self.vector_store.save_local(str(vector_path))
        logger.info(f"Vector store rebuilt with {len(memories)} entries")

    def save_memory(
        self,
        content: str,
        tags: Optional[List[str]] = None,
        importance: int = 5,
        context: Optional[str] = None
    ) -> Memory:
        """Save a new memory to the database and vector store."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                now = datetime.now(timezone.utc).isoformat()
                cur = conn.cursor()
                cur.execute(
                    """INSERT INTO memories 
                       (content, tags, created_at, importance, context, last_accessed) 
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (content, json.dumps(tags or []), now, importance, context, now)
                )
                conn.commit()
                memory_id = cur.lastrowid
                
            memory = Memory(
                id=memory_id,
                content=content,
                tags=(tags or []),
                created_at=now,
                importance=importance,
                context=context,
                last_accessed=now
            )

            # Update vector store
            if self.vector_store and self.embeddings:
                try:
                    self.vector_store.add_texts(
                        [content],
                        metadatas=[{"id": memory.id, "importance": importance, "tags": json.dumps(tags or [])}]
                    )
                    self.vector_store.save_local(VECTOR_STORE_PATH)
                except Exception as e:
                    logger.error(f"Failed to update vector store: {e}")

            logger.info(f"Saved memory #{memory_id}: {content[:50]}...")
            return memory

    def update_memory(
        self,
        memory_id: int,
        content: Optional[str] = None,
        tags: Optional[List[str]] = None,
        importance: Optional[int] = None
    ) -> Optional[Memory]:
        """Update an existing memory."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.cursor()
                
                # Get existing memory
                row = cur.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
                if not row:
                    return None
                
                memory = Memory.from_row(row)
                
                # Update fields
                if content is not None:
                    memory.content = content
                if tags is not None:
                    memory.tags = tags
                if importance is not None:
                    memory.importance = importance
                
                # Save updates
                cur.execute(
                    """UPDATE memories 
                       SET content = ?, tags = ?, importance = ? 
                       WHERE id = ?""",
                    (memory.content, json.dumps(memory.tags), memory.importance, memory_id)
                )
                conn.commit()
                
            # Rebuild vector store (simple approach)
            if self.embeddings:
                self._load_or_rebuild_vector_store()
            
            logger.info(f"Updated memory #{memory_id}")
            return memory

    def delete_memory(self, memory_id: int) -> bool:
        """Delete a memory and its associated reminders."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.cursor()
                cur.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
                deleted = cur.rowcount > 0
                conn.commit()
            
            if deleted and self.embeddings:
                self._load_or_rebuild_vector_store()
            
            logger.info(f"Deleted memory #{memory_id}")
            return deleted

    def semantic_search(self, query: str, limit: int = 5) -> List[Memory]:
        """Search memories using semantic similarity."""
        if not self.vector_store:
            logger.warning("Vector store not available, falling back to keyword search")
            return self._keyword_search(query, limit)
        
        try:
            results = self.vector_store.similarity_search_with_score(query, k=limit)
            if not results:
                return []
            
            # Get full memory objects
            ids = [r[0].metadata["id"] for r in results]
            with sqlite3.connect(self.db_path) as conn:
                placeholders = ','.join('?' * len(ids))
                rows = conn.execute(
                    f"SELECT * FROM memories WHERE id IN ({placeholders})",
                    ids
                ).fetchall()
                
                # Update last_accessed
                now = datetime.now(timezone.utc).isoformat()
                conn.execute(
                    f"UPDATE memories SET last_accessed = ? WHERE id IN ({placeholders})",
                    [now] + ids
                )
                conn.commit()
                
                memories = [Memory.from_row(r) for r in rows]
                
                # Sort by original order from vector search
                id_to_memory = {m.id: m for m in memories}
                return [id_to_memory[id] for id in ids if id in id_to_memory]
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return self._keyword_search(query, limit)

    def _keyword_search(self, query: str, limit: int = 5) -> List[Memory]:
        """Fallback keyword-based search."""
        with sqlite3.connect(self.db_path) as conn:
            query_pattern = f"%{query}%"
            rows = conn.execute(
                """SELECT * FROM memories 
                   WHERE content LIKE ? OR tags LIKE ? 
                   ORDER BY importance DESC, created_at DESC 
                   LIMIT ?""",
                (query_pattern, query_pattern, limit)
            ).fetchall()
            return [Memory.from_row(r) for r in rows]

    def get_memories_by_tag(self, tag: str, limit: int = 10) -> List[Memory]:
        """Get memories by tag."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """SELECT * FROM memories 
                   WHERE tags LIKE ? 
                   ORDER BY importance DESC, created_at DESC 
                   LIMIT ?""",
                (f'%"{tag}"%', limit)
            ).fetchall()
            return [Memory.from_row(r) for r in rows]

    def get_all_memories(self, order_by: str = "created_at DESC") -> List[Memory]:
        """Get all memories."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(f"SELECT * FROM memories ORDER BY {order_by}").fetchall()
            return [Memory.from_row(r) for r in rows]

    def set_reminder(
        self,
        memory_id: int,
        remind_at: datetime,
        recurrence: Optional[str] = None
    ) -> int:
        """Set a reminder for a memory."""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute(
                """INSERT INTO reminders 
                   (memory_id, remind_at, created_at, recurrence) 
                   VALUES (?, ?, ?, ?)""",
                (memory_id, remind_at.isoformat(), datetime.now(timezone.utc).isoformat(), recurrence)
            )
            conn.commit()
            reminder_id = cur.lastrowid
            logger.info(f"Set reminder #{reminder_id} for {remind_at}")
            return reminder_id

    def get_due_reminders(self) -> List[Tuple[Reminder, Memory]]:
        """Get all due reminders that haven't been completed."""
        with sqlite3.connect(self.db_path) as conn:
            now = datetime.now(timezone.utc).isoformat()
            rows = conn.execute(
                """SELECT r.*, m.* FROM reminders r 
                   JOIN memories m ON r.memory_id = m.id 
                   WHERE r.done = 0 AND r.remind_at <= ? 
                   ORDER BY r.remind_at ASC""",
                (now,)
            ).fetchall()
            return [(Reminder.from_row(row[:8]), Memory.from_row(row[8:])) for row in rows]

    def mark_reminder_done(self, reminder_id: int):
        """Mark a reminder as completed."""
        with sqlite3.connect(self.db_path) as conn:
            now = datetime.now(timezone.utc).isoformat()
            conn.execute(
                "UPDATE reminders SET done = 1, completed_at = ? WHERE id = ?",
                (now, reminder_id)
            )
            conn.commit()
            logger.info(f"Marked reminder #{reminder_id} as done")

    def snooze_reminder(self, reminder_id: int, minutes: int = 10):
        """Snooze a reminder for a specified duration."""
        with sqlite3.connect(self.db_path) as conn:
            snooze_until = (datetime.now(timezone.utc) + timedelta(minutes=minutes)).isoformat()
            conn.execute(
                "UPDATE reminders SET snoozed_until = ? WHERE id = ?",
                (snooze_until, reminder_id)
            )
            conn.commit()
            logger.info(f"Snoozed reminder #{reminder_id} for {minutes} minutes")

    def get_all_reminders(self, include_completed: bool = False) -> List[Tuple[Reminder, Memory]]:
        """Get all reminders."""
        with sqlite3.connect(self.db_path) as conn:
            query = """SELECT r.*, m.* FROM reminders r 
                       JOIN memories m ON r.memory_id = m.id"""
            if not include_completed:
                query += " WHERE r.done = 0"
            query += " ORDER BY r.remind_at ASC"
            
            rows = conn.execute(query).fetchall()
            return [(Reminder.from_row(row[:8]), Memory.from_row(row[8:])) for row in rows]

    def find_reminders_by_content(self, search_term: str) -> List[Tuple[Reminder, Memory]]:
        """Find reminders by searching their content."""
        with sqlite3.connect(self.db_path) as conn:
            search_pattern = f"%{search_term}%"
            rows = conn.execute(
                """SELECT r.*, m.* FROM reminders r 
                   JOIN memories m ON r.memory_id = m.id 
                   WHERE m.content LIKE ? AND r.done = 0
                   ORDER BY r.remind_at ASC""",
                (search_pattern,)
            ).fetchall()
            return [(Reminder.from_row(row[:8]), Memory.from_row(row[8:])) for row in rows]

    def delete_reminder(self, reminder_id: int) -> bool:
        """Delete a specific reminder by ID."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.cursor()
                cur.execute("DELETE FROM reminders WHERE id = ?", (reminder_id,))
                deleted = cur.rowcount > 0
                conn.commit()
            logger.info(f"Deleted reminder #{reminder_id}")
            return deleted

    def delete_all_reminders(self, include_completed: bool = True) -> int:
        """Delete all reminders. Returns count of deleted reminders."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.cursor()
                if include_completed:
                    cur.execute("DELETE FROM reminders")
                else:
                    cur.execute("DELETE FROM reminders WHERE done = 0")
                count = cur.rowcount
                conn.commit()
            logger.info(f"Deleted {count} reminders")
            return count

    def update_reminder(self, reminder_id: int, new_time: Optional[datetime] = None, new_memory_content: Optional[str] = None) -> Optional[Tuple[Reminder, Memory]]:
        """Update a reminder's time and/or content."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.cursor()
                
                # Get existing reminder
                row = cur.execute(
                    "SELECT r.*, m.* FROM reminders r JOIN memories m ON r.memory_id = m.id WHERE r.id = ?",
                    (reminder_id,)
                ).fetchone()
                
                if not row:
                    return None
                
                reminder = Reminder.from_row(row[:8])
                memory = Memory.from_row(row[8:])
                
                # Update reminder time if provided
                if new_time:
                    cur.execute(
                        "UPDATE reminders SET remind_at = ? WHERE id = ?",
                        (new_time.isoformat(), reminder_id)
                    )
                    reminder.remind_at = new_time.isoformat()
                
                # Update memory content if provided
                if new_memory_content:
                    cur.execute(
                        "UPDATE memories SET content = ? WHERE id = ?",
                        (new_memory_content, memory.id)
                    )
                    memory.content = new_memory_content
                
                conn.commit()
                logger.info(f"Updated reminder #{reminder_id}")
                return (reminder, memory)

    def get_statistics(self) -> Dict[str, Any]:
        """Get memory statistics."""
        with sqlite3.connect(self.db_path) as conn:
            stats = {}
            
            # Total memories
            stats['total_memories'] = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            
            # Total reminders
            stats['total_reminders'] = conn.execute("SELECT COUNT(*) FROM reminders").fetchone()[0]
            stats['pending_reminders'] = conn.execute("SELECT COUNT(*) FROM reminders WHERE done = 0").fetchone()[0]
            
            # Average importance
            result = conn.execute("SELECT AVG(importance) FROM memories").fetchone()[0]
            stats['avg_importance'] = round(result, 2) if result else 0
            
            # Most common tags
            all_tags = []
            rows = conn.execute("SELECT tags FROM memories WHERE tags IS NOT NULL").fetchall()
            for row in rows:
                all_tags.extend(json.loads(row[0]))
            
            tag_counts = defaultdict(int)
            for tag in all_tags:
                tag_counts[tag] += 1
            
            stats['top_tags'] = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            
            return stats


# ============================================================================
# Time Expression Parser
# ============================================================================
class TimeExpressionParser:
    """Parse natural language time expressions into datetime objects using LLM."""
    
    def __init__(self, llm=None):
        self.llm = llm
    
    def smart_parse(self, expr: str, reference_time: Optional[datetime] = None) -> datetime:
        """Parse time expression using LLM only."""
        if not self.llm:
            raise ValueError("LLM required for time parsing")
            
        if reference_time is None:
            reference_time = datetime.now(timezone.utc)
        
        expr = expr.lower().strip()
        
        try:
            prompt = f"""Parse this natural language time expression into an ISO 8601 datetime string (YYYY-MM-DDTHH:MM:SS).
Current time: {reference_time.strftime('%Y-%m-%d %H:%M:%S')} UTC
Expression: "{expr}"

CRITICAL: Return ONLY the ISO datetime string with NO explanation, NO quotes, NO extra text.
Examples:
"in 5 minutes" -> {(reference_time + timedelta(minutes=5)).strftime('%Y-%m-%dT%H:%M:%S')}
"tomorrow at 3pm" -> {(reference_time + timedelta(days=1)).replace(hour=15, minute=0, second=0).strftime('%Y-%m-%dT%H:%M:%S')}
"in 2 hours" -> {(reference_time + timedelta(hours=2)).strftime('%Y-%m-%dT%H:%M:%S')}
"next week" -> {(reference_time + timedelta(days=7)).replace(hour=9, minute=0, second=0).strftime('%Y-%m-%dT%H:%M:%S')}
"in one min" -> {(reference_time + timedelta(minutes=1)).strftime('%Y-%m-%dT%H:%M:%S')}
"tomorrow" -> {(reference_time + timedelta(days=1)).replace(hour=9, minute=0, second=0).strftime('%Y-%m-%dT%H:%M:%S')}

Return ONLY the datetime string:"""
            messages = [{"role": "user", "content": prompt}]
            response = self.llm.invoke(messages)
            result = response.content.strip()
            
            # Extract ISO datetime string from response (LLM might add extra text)
            import re
            iso_pattern = r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}'
            match = re.search(iso_pattern, result)
            if match:
                result = match.group(0)
            
            if result and result != "UNPARSEABLE":
                # Try to parse the result
                try:
                    parsed = datetime.fromisoformat(result.replace('Z', '+00:00'))
                    if parsed.tzinfo is None:
                        parsed = parsed.replace(tzinfo=timezone.utc)
                    logger.info(f"LLM parsed '{expr}' as {parsed}")
                    return parsed
                except ValueError:
                    logger.warning(f"LLM returned invalid datetime: {result}")
                    raise ValueError(f"Could not parse time expression: {expr}")
            else:
                raise ValueError(f"LLM could not parse time expression: {expr}")
        except Exception as e:
            logger.error(f"LLM parsing failed for '{expr}': {e}")
            raise ValueError(f"Failed to parse time expression: {expr}")


# ============================================================================
# Reminder Daemon
# ============================================================================
class ReminderDaemon:
    """Background daemon for checking and notifying reminders."""
    
    def __init__(self, store: EnhancedMemoryStore, console: Console):
        self.store = store
        self.console = console
        self.toaster = None  # Disabled due to Windows compatibility issues
        self.running = False
        self.thread = None

    def start(self):
        """Start the reminder daemon."""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._check_loop, daemon=True)
        self.thread.start()
        logger.info("Reminder daemon started")

    def stop(self):
        """Stop the reminder daemon."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        logger.info("Reminder daemon stopped")

    def _check_loop(self):
        """Main loop for checking reminders."""
        while self.running:
            try:
                due_reminders = self.store.get_due_reminders()
                
                for reminder, memory in due_reminders:
                    self._notify(reminder, memory)
                    self.store.mark_reminder_done(reminder.id)
                    
            except Exception as e:
                logger.error(f"Reminder check error: {e}", exc_info=True)
            
            time.sleep(REMINDER_CHECK_INTERVAL)

    def _notify(self, reminder: Reminder, memory: Memory):
        """Send notification for a reminder."""
        # Console notification
        if RICH_AVAILABLE:
            self.console.print(
                Panel(
                    f"[bold yellow]⏰ REMINDER[/bold yellow]\n\n{memory.content}",
                    title=f"Reminder #{reminder.id}",
                    border_style="yellow"
                )
            )
        else:
            print(f"\n⏰ REMINDER: {memory.content}\n")
        
        # Desktop notification (Windows)
        if self.toaster:
            try:
                self.toaster.show_toast(
                    "AI Assistant Reminder",
                    memory.content[:250],  # Truncate if too long
                    duration=10,
                    threaded=True
                )
            except Exception as e:
                logger.warning(f"Failed to show desktop notification: {e}")
        
        # Audio alert (Windows)
        if WINDOWS_NOTIFICATIONS_AVAILABLE:
            try:
                winsound.MessageBeep(winsound.MB_ICONASTERISK)
            except Exception as e:
                logger.warning(f"Failed to play audio alert: {e}")


# ============================================================================
# Advanced Chat Agent
# ============================================================================
class AdvancedChatAgent:
    """Main chat agent with memory and tool capabilities."""
    
    def __init__(self, model_id: str = DEFAULT_MODEL):
        self.store = EnhancedMemoryStore()
        self.console = Console() if RICH_AVAILABLE else Console()
        self.model_id = model_id
        self.llm = self._init_llm(model_id=self.model_id)
        self.tools = self._create_tools()
        self.memory = ChatMessageHistory()
        self.reminder_daemon = ReminderDaemon(self.store, self.console)
        self.conversation_history = []
        self.time_parser = TimeExpressionParser(self.llm)
        logger.info(f"Agent initialized with model: {self.model_id}")

    def _init_llm(self, model_id: str) -> HFChatLLM:
        """Initialize the LLM with API key validation."""
        api_key = os.environ.get("HF_API_KEY") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        if not api_key:
            logger.error("HF_API_KEY not found in environment variables")
            self.console.print("[bold red]Error: HF_API_KEY not found. Please set it in .env file.[/bold red]")
            sys.exit(1)
        
        client = InferenceClient(token=api_key)
        return HFChatLLM(client=client, model=model_id, max_retries=3)

    def _create_tools(self) -> List[StructuredTool]:
        """Create tool definitions for the agent."""
        
        def save_memory_tool(content: str, tags: Optional[List[str]] = None, importance: int = 5, context: Optional[str] = None) -> str:
            """Save important information to memory."""
            try:
                memory = self.store.save_memory(content, tags=tags, importance=importance, context=context)
                tag_str = f" (tags: {', '.join(memory.tags)})" if memory.tags else ""
                return f"✓ Memory saved #{memory.id}{tag_str}: {content[:60]}{'...' if len(content) > 60 else ''}"
            except Exception as e:
                logger.error(f"Error saving memory: {e}")
                return f"❌ Failed to save memory: {str(e)}"

        def find_memories_tool(query: str = None, search_term: str = None, limit: int = 5) -> str:
            """Search for relevant memories."""
            try:
                # Handle both parameter names
                search_query = query or search_term
                if not search_query:
                    return "Please provide a search query."
                
                memories = self.store.semantic_search(search_query, limit)
                if not memories:
                    return f"No memories found matching '{search_query}'."
                
                results = [f"Found {len(memories)} relevant memories:\n"]
                for i, m in enumerate(memories, 1):
                    tag_str = f" [{', '.join(m.tags)}]" if m.tags else ""
                    importance_stars = "⭐" * min(m.importance, 10)
                    content_preview = m.content[:100] + "..." if len(m.content) > 100 else m.content
                    results.append(f"{i}. #{m.id} {importance_stars}{tag_str}\n   {content_preview}")
                
                return "\n".join(results)
            except Exception as e:
                logger.error(f"Error finding memories: {e}")
                return f"❌ Failed to search memories: {str(e)}"

        def set_reminder_tool(
            memory_content: str = None,
            when: str = None,
            description: str = None,
            time: str = None,
            content: str = None
        ) -> str:
            """Set a reminder for the future."""
            try:
                # Handle different parameter formats
                reminder_content = memory_content or description or content
                time_str = when or time
                
                if not reminder_content:
                    return "❌ Please provide what you want to be reminded about."
                if not time_str:
                    return "❌ Please specify when you want to be reminded."
                
                # Parse time expression using smart LLM parsing
                remind_at = self.time_parser.smart_parse(time_str)
                logger.info(f"Parsed '{time_str}' as {remind_at}")
                
                # Validate future time
                now = datetime.now(timezone.utc)
                if remind_at <= now:
                    return f"❌ Cannot set reminder in the past. '{time_str}' resolves to {remind_at.strftime('%Y-%m-%d %H:%M')} UTC which has already passed."
                
                # Check if the parsed time seems reasonable (not too far in the future)
                time_diff = remind_at - now
                if time_diff.days > 365:  # More than a year
                    return f"⚠️ That seems very far in the future. '{time_str}' resolves to {remind_at.strftime('%Y-%m-%d %H:%M')} UTC. Are you sure this is correct?"
                
                # Warn if parsing might have failed (defaulted to 1 hour)
                if time_diff.seconds >= 3599 and time_diff.seconds <= 3601:  # Approximately 1 hour
                    logger.warning(f"Time parsing may have failed for '{time_str}', defaulted to 1 hour")
                    return f"⚠️ I didn't understand '{time_str}' clearly, so I set it for 1 hour from now. Try formats like: 'in 5 minutes', 'tomorrow at 3pm', 'in 2 hours'.\n   Reminder: {reminder_content}"
                
                # Save memory and set reminder
                memory = self.store.save_memory(reminder_content, tags=["reminder"])
                self.store.set_reminder(memory.id, remind_at)
                
                # Format response
                time_diff = remind_at - now
                if time_diff.days > 0:
                    time_desc = f"{time_diff.days} day{'s' if time_diff.days > 1 else ''}"
                else:
                    hours = time_diff.seconds // 3600
                    minutes = (time_diff.seconds % 3600) // 60
                    if hours > 0:
                        time_desc = f"{hours} hour{'s' if hours > 1 else ''} and {minutes} minute{'s' if minutes > 1 else ''}"
                    else:
                        time_desc = f"{minutes} minute{'s' if minutes > 1 else ''}"
                
                # Convert to local time for display
                local_time = remind_at.replace(tzinfo=timezone.utc).astimezone()
                local_time_str = local_time.strftime('%Y-%m-%d %I:%M %p')
                return f"✓ Reminder set for {local_time_str} (in {time_desc})\n   {reminder_content}"
            
            except Exception as e:
                logger.error(f"Error setting reminder: {e}")
                return f"❌ Failed to set reminder: {str(e)}"

        def show_reminders_tool(show_completed: bool = False) -> str:
            """Show all reminders with status indicators."""
            try:
                reminders = self.store.get_all_reminders(include_completed=show_completed)
                
                if not reminders:
                    return "No reminders found." if not show_completed else "No reminders found (checked completed too)."
                
                now_utc = datetime.now(timezone.utc)
                now_local = datetime.now()  # Local timezone
                pending = []
                overdue = []
                completed = []
                
                for reminder, memory in reminders:
                    remind_time_utc = datetime.fromisoformat(reminder.remind_at.replace('Z', '+00:00'))
                    remind_time_local = remind_time_utc.replace(tzinfo=timezone.utc).astimezone()  # Convert to local
                    
                    if reminder.done:
                        completed.append((remind_time_utc, remind_time_local, reminder, memory))
                    elif remind_time_utc < now_utc:
                        overdue.append((remind_time_utc, remind_time_local, reminder, memory))
                    else:
                        pending.append((remind_time_utc, remind_time_local, reminder, memory))
                
                # Sort by time
                pending.sort()
                overdue.sort()
                completed.sort(reverse=True)
                
                result = []
                
                # Overdue reminders (highest priority)
                if overdue:
                    result.append("⚠️  OVERDUE REMINDERS:")
                    for remind_time_utc, remind_time_local, reminder, memory in overdue:
                        time_str = remind_time_local.strftime('%Y-%m-%d %H:%M')
                        time_ago = now_utc - remind_time_utc
                        if time_ago.days > 0:
                            ago_str = f"{time_ago.days} day{'s' if time_ago.days > 1 else ''} ago"
                        else:
                            hours_ago = time_ago.seconds // 3600
                            ago_str = f"{hours_ago} hour{'s' if hours_ago > 1 else ''} ago"
                        result.append(f"   • [ID: {reminder.id}] {memory.content}")
                        result.append(f"     Was due: {time_str} ({ago_str})")
                
                # Upcoming reminders
                if pending:
                    if result:
                        result.append("")
                    result.append("⏳ UPCOMING REMINDERS:")
                    for remind_time_utc, remind_time_local, reminder, memory in pending:
                        time_str = remind_time_local.strftime('%Y-%m-%d %H:%M')
                        time_until = remind_time_utc - now_utc
                        if time_until.days > 0:
                            until_str = f"in {time_until.days} day{'s' if time_until.days > 1 else ''}"
                        else:
                            hours_until = time_until.seconds // 3600
                            minutes_until = (time_until.seconds % 3600) // 60
                            if hours_until > 0:
                                until_str = f"in {hours_until}h {minutes_until}m"
                            else:
                                until_str = f"in {minutes_until} minute{'s' if minutes_until > 1 else ''}"
                        result.append(f"   • [ID: {reminder.id}] {memory.content}")
                        result.append(f"     Scheduled: {time_str} ({until_str})")
                
                # Completed reminders
                if show_completed and completed:
                    if result:
                        result.append("")
                    result.append("✅ COMPLETED REMINDERS:")
                    for remind_time_utc, remind_time_local, reminder, memory in completed[:5]:  # Show only last 5
                        time_str = remind_time_local.strftime('%Y-%m-%d %H:%M')
                        result.append(f"   • [ID: {reminder.id}] {memory.content}")
                        result.append(f"     Was scheduled: {time_str}")
                    if len(completed) > 5:
                        result.append(f"   ... and {len(completed) - 5} more completed")
                
                return "\n".join(result)
            
            except Exception as e:
                logger.error(f"Error showing reminders: {e}")
                return f"❌ Failed to show reminders: {str(e)}"

        def update_memory_tool(memory_id: int, content: Optional[str] = None, tags: Optional[List[str]] = None, importance: Optional[int] = None) -> str:
            """Update an existing memory."""
            try:
                memory = self.store.update_memory(memory_id, content, tags, importance)
                if not memory:
                    return f"❌ Memory #{memory_id} not found."
                return f"✓ Updated memory #{memory_id}"
            except Exception as e:
                logger.error(f"Error updating memory: {e}")
                return f"❌ Failed to update memory: {str(e)}"

        def delete_memory_tool(memory_id: int) -> str:
            """Delete a memory."""
            try:
                if self.store.delete_memory(memory_id):
                    return f"✓ Deleted memory #{memory_id}"
                return f"❌ Memory #{memory_id} not found."
            except Exception as e:
                logger.error(f"Error deleting memory: {e}")
                return f"❌ Failed to delete memory: {str(e)}"

        def delete_reminder_tool(
            reminder_id: Optional[int] = None, 
            reminder_ids: Optional[List[int]] = None,
            ids: Optional[List[int]] = None,
            id: Optional[int] = None,
            delete_all: bool = False, 
            all: bool = False, 
            include_completed: bool = True
        ) -> str:
            """Delete one or all reminders."""
            try:
                # Handle both parameter names for delete_all
                should_delete_all = delete_all or all
                
                # Handle multiple IDs - check all possible parameter names
                ids_to_delete = reminder_ids or ids
                
                # Handle single ID - check all possible parameter names
                single_id = reminder_id or id
                
                if should_delete_all:
                    count = self.store.delete_all_reminders(include_completed=include_completed)
                    if count == 0:
                        return "No reminders to delete."
                    reminder_type = "reminders" if include_completed else "pending reminders"
                    return f"✓ Deleted {count} {reminder_type}"
                elif ids_to_delete:
                    # Delete multiple reminders
                    deleted = []
                    not_found = []
                    for rid in ids_to_delete:
                        if self.store.delete_reminder(rid):
                            deleted.append(rid)
                        else:
                            not_found.append(rid)
                    
                    results = []
                    if deleted:
                        results.append(f"✓ Deleted reminder(s): {', '.join(f'#{id}' for id in deleted)}")
                    if not_found:
                        results.append(f"❌ Not found: {', '.join(f'#{id}' for id in not_found)}")
                    return "\n".join(results)
                elif single_id is not None:
                    if self.store.delete_reminder(single_id):
                        return f"✓ Deleted reminder #{single_id}"
                    return f"❌ Reminder #{single_id} not found."
                else:
                    return "❌ Please specify either a reminder_id, reminder_ids, or set delete_all=True"
            except Exception as e:
                logger.error(f"Error deleting reminder: {e}", exc_info=True)
                return f"❌ Failed to delete reminder: {str(e)}"

        def update_reminder_tool(reminder_id: int, new_time: Optional[str] = None, new_content: Optional[str] = None) -> str:
            """Update a reminder's time and/or content."""
            try:
                # Parse new time if provided
                new_datetime = None
                if new_time:
                    new_datetime = self.time_parser.smart_parse(new_time)
                    # Validate future time
                    now = datetime.now(timezone.utc)
                    if new_datetime <= now:
                        return f"❌ Cannot set reminder in the past. '{new_time}' resolves to {new_datetime.strftime('%Y-%m-%d %H:%M')} UTC"
                
                # Update reminder
                result = self.store.update_reminder(reminder_id, new_datetime, new_content)
                if not result:
                    return f"❌ Reminder #{reminder_id} not found."
                
                reminder, memory = result
                
                # Build response
                updates = []
                if new_datetime:
                    local_time = new_datetime.replace(tzinfo=timezone.utc).astimezone()
                    updates.append(f"time changed to {local_time.strftime('%Y-%m-%d %H:%M')}")
                if new_content:
                    updates.append(f"content changed to '{new_content}'")
                
                return f"✓ Updated reminder #{reminder_id}: {', '.join(updates)}"
            except Exception as e:
                logger.error(f"Error updating reminder: {e}")
                return f"❌ Failed to update reminder: {str(e)}"

        def find_reminders_tool(query: str = None, search_term: str = None, limit: int = 5) -> str:
            """Find reminders by searching their content."""
            try:
                # Handle both parameter names
                search_query = query or search_term
                if not search_query:
                    return "Please provide a search query."
                
                reminders = self.store.find_reminders_by_content(search_query)
                if not reminders:
                    return f"No reminders found matching '{search_query}'."
                
                # Limit results if specified
                if limit and len(reminders) > limit:
                    reminders = reminders[:limit]
                
                now_utc = datetime.now(timezone.utc)
                now_local = datetime.now()  # Local timezone
                pending = []
                overdue = []
                
                for reminder, memory in reminders:
                    remind_time_utc = datetime.fromisoformat(reminder.remind_at.replace('Z', '+00:00'))
                    remind_time_local = remind_time_utc.replace(tzinfo=timezone.utc).astimezone()  # Convert to local
                    
                    if remind_time_utc < now_utc:
                        overdue.append((remind_time_utc, remind_time_local, reminder, memory))
                    else:
                        pending.append((remind_time_utc, remind_time_local, reminder, memory))
                
                # Sort by time
                pending.sort()
                overdue.sort()
                
                result = [f"Found {len(reminders)} reminder(s) matching '{search_query}':\n"]
                
                # Overdue reminders (highest priority)
                if overdue:
                    result.append("⚠️  OVERDUE REMINDERS:")
                    for remind_time_utc, remind_time_local, reminder, memory in overdue:
                        time_str = remind_time_local.strftime('%Y-%m-%d %H:%M')
                        time_ago = now_utc - remind_time_utc
                        if time_ago.days > 0:
                            ago_str = f"{time_ago.days} day{'s' if time_ago.days > 1 else ''} ago"
                        else:
                            hours_ago = time_ago.seconds // 3600
                            ago_str = f"{hours_ago} hour{'s' if hours_ago > 1 else ''} ago"
                        result.append(f"   • [ID: {reminder.id}] {memory.content}")
                        result.append(f"     Was due: {time_str} ({ago_str})")
                
                # Upcoming reminders
                if pending:
                    if result and len(result) > 1:
                        result.append("")
                    result.append("⏳ UPCOMING REMINDERS:")
                    for remind_time_utc, remind_time_local, reminder, memory in pending:
                        time_str = remind_time_local.strftime('%Y-%m-%d %H:%M')
                        time_until = remind_time_utc - now_utc
                        if time_until.days > 0:
                            until_str = f"in {time_until.days} day{'s' if time_until.days > 1 else ''}"
                        else:
                            hours_until = time_until.seconds // 3600
                            minutes_until = (time_until.seconds % 3600) // 60
                            if hours_until > 0:
                                until_str = f"in {hours_until}h {minutes_until}m"
                            else:
                                until_str = f"in {minutes_until} minute{'s' if minutes_until > 1 else ''}"
                        result.append(f"   • [ID: {reminder.id}] {memory.content}")
                        result.append(f"     Scheduled: {time_str} ({until_str})")
                
                return "\n".join(result)
            
            except Exception as e:
                logger.error(f"Error finding reminders: {e}")
                return f"❌ Failed to search reminders: {str(e)}"

        def show_statistics_tool() -> str:
            """Show memory and reminder statistics."""
            try:
                stats = self.store.get_statistics()
                result = [
                    "📊 Memory Statistics:",
                    f"   • Total memories: {stats['total_memories']}",
                    f"   • Average importance: {stats['avg_importance']}/10",
                    f"   • Total reminders: {stats['total_reminders']}",
                    f"   • Pending reminders: {stats['pending_reminders']}",
                ]
                
                if stats['top_tags']:
                    result.append("\n   Top tags:")
                    for tag, count in stats['top_tags']:
                        result.append(f"     - {tag}: {count}")
                
                return "\n".join(result)
            except Exception as e:
                logger.error(f"Error showing statistics: {e}")
                return f"❌ Failed to show statistics: {str(e)}"

        return [
            StructuredTool.from_function(
                func=save_memory_tool,
                name="save_memory",
                description="Save important information, facts, or preferences to long-term memory. Use proactively when user shares important details.",
                args_schema=SaveMemoryInput
            ),
            StructuredTool.from_function(
                func=find_memories_tool,
                name="find_memories",
                description="Search saved memories using semantic search. Use when user asks about past conversations or information.",
                args_schema=FindMemoriesInput
            ),
            StructuredTool.from_function(
                func=set_reminder_tool,
                name="set_reminder",
                description="Set a reminder for the future. Use when user mentions future tasks, events, or deadlines.",
                args_schema=SetReminderInput
            ),
            StructuredTool.from_function(
                func=show_reminders_tool,
                name="show_reminders",
                description="CRITICAL: Use immediately when user asks about reminders, schedule, calendar, appointments, or timers. Shows all reminders with status. Do NOT use for searching specific reminders.",
                args_schema=ShowRemindersInput
            ),
            StructuredTool.from_function(
                func=update_memory_tool,
                name="update_memory",
                description="Update an existing memory's content, tags, or importance.",
                args_schema=UpdateMemoryInput
            ),
            StructuredTool.from_function(
                func=delete_memory_tool,
                name="delete_memory",
                description="Delete a specific memory by ID.",
                args_schema=DeleteMemoryInput
            ),
            StructuredTool.from_function(
                func=delete_reminder_tool,
                name="delete_reminder",
                description="Delete one, multiple, or all reminders. ALWAYS use this tool when user asks to delete/remove/clear reminders. Supports single ID, multiple IDs, or delete_all flag.",
                args_schema=DeleteReminderInput
            ),
            StructuredTool.from_function(
                func=update_reminder_tool,
                name="update_reminder",
                description="Update/modify a reminder's time or content. Use when user asks to change/update/modify a reminder.",
                args_schema=UpdateReminderInput
            ),
            StructuredTool.from_function(
                func=find_reminders_tool,
                name="find_reminders",
                description="Find reminders by searching their content with specific keywords or phrases. Only use when user wants to search for reminders containing particular words, not for general reminder viewing.",
                args_schema=FindRemindersInput
            ),
            StructuredTool.from_function(
                func=show_statistics_tool,
                name="show_statistics",
                description="Show statistics about memories and reminders."
            ),
        ]

    def _get_system_prompt(self) -> str:
        """Get the comprehensive system prompt."""
        tools_desc = []
        for i, tool in enumerate(self.tools, 1):
            tools_desc.append(f"{i}. **{tool.name}**: {tool.description}")
        
        current_time = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
        
        prompt = f"""You are an advanced AI assistant with perfect memory and proactive intelligence.

CURRENT TIME: {current_time}

CAPABILITIES:
- Long-term memory storage and semantic recall
- Proactive reminder system with time parsing
- Natural conversation with context awareness
- Intelligent information organization

AVAILABLE TOOLS:
{chr(10).join(tools_desc)}

BEHAVIORAL GUIDELINES:

1. **Proactive Memory**
   - AUTOMATICALLY save important info (names, preferences, projects, deadlines) without asking
   - Use importance scores: 9-10 (critical), 7-8 (important), 5-6 (useful), 3-4 (casual)
   - Add relevant tags for better organization

2. **Smart Search**
   - AUTOMATICALLY search memory when questions relate to past conversations
   - Use semantic search to find relevant context

3. **Reminder Intelligence**
   - OFFER to set reminders when users mention future tasks
   - Parse natural time expressions correctly
   - When asked about schedule/calendar/reminders/timers, use show_reminders immediately
   - Only use find_reminders when user wants to search for specific reminders by keywords

4. **Natural Interaction**
   - Be conversational and friendly
   - Don't announce internal actions ("I'm saving this...")
   - Let tool outputs confirm actions
   - Keep responses concise and relevant

5. **Response Format**
   - ALWAYS respond with ONLY valid JSON
   - No text before or after the JSON object
   - Structure: {{"response": "text", "tool_call": {{"name": "tool", "args": {{}}}}}}
   - Omit "tool_call" if no tool needed

EXAMPLES (respond with ONLY JSON):
- Simple: {{"response": "Hello! I'm here to help you stay organized and remember important things."}}
- With tool: {{"response": "Let me check your upcoming reminders.", "tool_call": {{"name": "show_reminders", "args": {{"show_completed": false}}}}}}
- Proactive: {{"response": "That sounds like an important deadline!", "tool_call": {{"name": "save_memory", "args": {{"content": "Project X deadline on Friday", "tags": ["deadline", "work"], "importance": 9}}}}}}

Remember: Be helpful, proactive, and natural. Your goal is to be an indispensable memory companion."""
        
        return prompt

    def _parse_response(self, response_text: str) -> Tuple[str, Optional[str], Optional[dict]]:
        """Parse JSON response from model."""
        response_text = response_text.strip()
        
        # Find JSON object
        start_idx = response_text.find('{')
        if start_idx == -1:
            logger.warning(f"No JSON found in response: {response_text[:200]}")
            return response_text, None, None
        
        # Find matching closing brace
        brace_count = 0
        end_idx = -1
        for i in range(start_idx, len(response_text)):
            if response_text[i] == '{':
                brace_count += 1
            elif response_text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i
                    break
        
        if end_idx == -1:
            logger.warning(f"No matching closing brace found")
            return response_text, None, None
        
        json_str = response_text[start_idx:end_idx+1]
        try:
            data = json.loads(json_str)
            response_msg = data.get("response", "")
            tool_call = data.get("tool_call")
            
            if tool_call and isinstance(tool_call, dict):
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("args", {})
                return response_msg, tool_name, tool_args
            
            return response_msg, None, None
        
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e} | JSON string: {json_str}")
            # Fallback to plain text
            return response_text, None, None

    def _execute_tool(self, tool_name: str, args: dict) -> str:
        """Execute a tool by name."""
        for tool in self.tools:
            if tool.name == tool_name:
                try:
                    # Validate arguments against the tool's schema if available
                    if hasattr(tool, 'args_schema') and tool.args_schema:
                        # Filter out invalid arguments
                        valid_args = {}
                        schema_fields = tool.args_schema.model_fields if hasattr(tool.args_schema, 'model_fields') else (tool.args_schema.__fields__ if hasattr(tool.args_schema, '__fields__') else {})
                        for arg_name, arg_value in args.items():
                            if arg_name in schema_fields:
                                valid_args[arg_name] = arg_value
                            else:
                                logger.warning(f"Tool {tool_name} received invalid argument '{arg_name}', ignoring")
                        
                        result = tool.func(**valid_args)
                    else:
                        result = tool.func(**args)
                    return str(result)
                except Exception as e:
                    logger.error(f"Tool execution error ({tool_name}): {e}", exc_info=True)
                    return f"❌ Error executing {tool_name}: {str(e)}"
        
        return f"❌ Unknown tool: {tool_name}"

    def chat(self):
        """Main chat loop."""
        # Start reminder daemon
        self.reminder_daemon.start()
        
        # Welcome message
        if RICH_AVAILABLE:
            self.console.print(
                Panel(
                    f"[bold cyan]Advanced AI Memory Assistant[/bold cyan]\n"
                    f"Model: [white]{self.model_id}[/white]\n\n"
                    f"I have perfect memory and can set reminders for you.\n"
                    f"Type 'quit', 'exit', or 'bye' to end the conversation.",
                    title="Welcome",
                    border_style="cyan"
                )
            )
        else:
            print(f"\n{'='*60}")
            print(f"Advanced AI Memory Assistant")
            print(f"Model: {self.model_id}")
            print(f"{'='*60}\n")
        
        system_prompt = self._get_system_prompt()
        
        while True:
            try:
                # Get user input
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                user_input = Prompt.ask(f"\n[dim]{current_time}[/dim] [bold green]You[/bold green]") if RICH_AVAILABLE else input(f"\n[{current_time}] You: ")
                
                if user_input.lower().strip() in ("quit", "exit", "bye", "goodbye"):
                    break
                
                if not user_input.strip():
                    continue
                
                # Build conversation context (last 5 exchanges)
                context = ""
                if self.conversation_history:
                    recent = self.conversation_history[-10:]  # Last 5 exchanges (10 messages)
                    context = "\n".join([f"{msg['role'].title()}: {msg['content']}" for msg in recent])
                    context = f"\n\nRECENT CONVERSATION:\n{context}\n"
                
                # Create prompt
                full_prompt = f"{system_prompt}{context}\nUser: {user_input}\n\nAssistant:"
                
                # Get response
                messages = [{"role": "user", "content": full_prompt}]
                response = self.llm.invoke(messages)
                model_output = response.content.strip()
                
                # Parse response
                response_text, tool_name, tool_args = self._parse_response(model_output)
                
                # Execute tool if needed
                tool_result = None
                if tool_name and tool_args:
                    tool_result = self._execute_tool(tool_name, tool_args)
                
                # Display response
                response_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                if RICH_AVAILABLE:
                    self.console.print(f"[dim]{response_time}[/dim] ", end="")
                    if response_text:
                        self.console.print(Markdown(f"**Assistant**: {response_text}"))
                    if tool_result:
                        self.console.print(f"\n{tool_result}")
                    if not response_text and not tool_result:
                        self.console.print(Markdown("**Assistant**: Done!"))
                else:
                    print(f"\n[{response_time}] Assistant: {response_text}")
                    if tool_result:
                        print(tool_result)
                
                # Update conversation history
                self.conversation_history.append({"role": "user", "content": user_input})
                self.conversation_history.append({"role": "assistant", "content": response_text or tool_result or "Done"})
                
                # Trim history to last 20 messages
                if len(self.conversation_history) > 20:
                    self.conversation_history = self.conversation_history[-20:]
                
            except (KeyboardInterrupt, EOFError):
                print()
                break
            except Exception as e:
                logger.error(f"Chat error: {e}", exc_info=True)
                if RICH_AVAILABLE:
                    self.console.print(f"[bold red]Error: {str(e)}[/bold red]")
                else:
                    print(f"Error: {str(e)}")
        
        # Cleanup
        self.reminder_daemon.stop()
        if RICH_AVAILABLE:
            self.console.print("\n[yellow]Goodbye! All your memories are safely stored.[/yellow]")
        else:
            print("\nGoodbye! All your memories are safely stored.")


# ============================================================================
# Main Entry Point
# ============================================================================
def main():
    """Main entry point for the application."""
    try:
        # Get model from environment or use default
        model_to_use = os.environ.get("MODEL", DEFAULT_MODEL)
        
        # Initialize and run agent
        agent = AdvancedChatAgent(model_id=model_to_use)
        agent.chat()
        
    except Exception as e:
        logger.critical(f"Fatal initialization error: {e}", exc_info=True)
        print(f"\nFatal Error: {str(e)}")
        print("Please check your configuration and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()