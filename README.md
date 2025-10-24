# Advanced AI Memory Agent

![AI Memory Agent Interface](images/Képernyőkép%202025-10-24%20204521.png)

**A persistent, proactive, and conversational AI agent designed to be a true digital companion. Powered by LangChain and state-of-the-art Hugging Face models.**

This project addresses the fundamental limitation of most chatbots: their lack of memory. The Advanced AI Memory Agent is built from the ground up to remember, recall, and act upon information from past conversations. It uses a sophisticated dual-memory system to maintain both short-term context and long-term semantic knowledge, enabling it to function as a reliable, stateful assistant.

---

## Why This Agent Matters

In a world of stateless AI, this agent provides:
*   **Continuity:** Pick up conversations right where you left off, days or weeks later.
*   **Context:** The agent leverages past information to better understand your current needs.
*   **Proactivity:** It doesn't just respond; it anticipates needs, offering to save key information and set reminders.
*   **Personalization:** The longer you use it, the more it learns about your projects, preferences, and important details, becoming a truly personalized tool.

---

## Core Features Explained

-   🧠 **Semantic Long-Term Memory**: The agent uses a FAISS vector database to store memories. When you ask a question, it searches for information based on **conceptual meaning**, not just exact keyword matches. This allows it to find the "spirit" of what you're asking for, even if you phrase it differently.

-   🛠️ **Autonomous Tool Use (ReAct Framework)**: This is the agent's "mind." It follows a **Reason-Act** loop. When you give it a prompt, it first "thinks" about what it needs to do. This thought process might lead it to conclude that it should use one of its special tools, like saving a memory or setting a reminder.

-   🗄️ **Persistent & Queryable Storage**: All memories and reminders are stored permanently in a local **SQLite database**. This data is structured and durable, ensuring that your agent's knowledge base is safe and sound between sessions.

-   ⏰ **Natural Language Smart Reminders**: You can ask the agent to "remind me to call John tomorrow morning" or "set a reminder in 3 hours." It parses these natural language requests, saves them as a high-importance memory, and runs a background process to notify you at the correct time.

-   � **Windows Notifications & Audio Alerts**: On Windows systems, reminders trigger both desktop toast notifications and audio alerts in addition to console notifications, ensuring you never miss important reminders.

-   �🔌 **Dynamic Model Backend**: The agent is not tied to a single model. You can easily switch the underlying Hugging Face model via an environment variable, allowing you to experiment with different LLMs (like Mistral, Gemma, Llama, etc.) without changing a single line of code.

-   🖥️ **Enhanced User Experience**: The command-line interface is powered by the `rich` library, providing formatted text, colors, and clear panels for a more pleasant and readable interaction.

---

## Architecture Deep Dive

The agent operates on a modular architecture orchestrated by LangChain.

```mermaid
graph TD
    subgraph User Interaction
        A[User Input via CLI]
    end

    subgraph Agent Core [The Brain: AgentExecutor with ReAct Loop]
        B[AgentExecutor] -- Manages Flow --> B
        B -- "What should I do next?" --> C{LLM (Hugging Face)}
        C -- "Thought: I should save this memory." --> B
        B -- "Use save_memory tool" --> T1[save_memory]
    end

    subgraph Agent Capabilities [The Skills: Tools & Memory]
        subgraph Tools
            T1 -- Writes to --> M_LT
            T2[find_memories] -- Reads from --> M_LT
            T3[set_reminder] -- Writes to --> M_LT
        end
        subgraph Memory
            M_ST[Short-Term: ConversationBufferWindowMemory]
            M_LT[Long-Term: EnhancedMemoryStore]
        end
    end
    
    subgraph Data Persistence Layer
        DB[(SQLite Database)]
        VDB[(FAISS Vector Store)]
    end

    A --> B
    B -- Maintains Context --> M_ST
    M_LT --> DB & VDB
```
1.  **The User Input:** You provide a prompt through the CLI.
2.  **The Brain (AgentExecutor):** This is the central loop. It takes your input, combines it with the recent conversation history (from Short-Term Memory), and formats it into a detailed prompt for the LLM.
3.  **The LLM (Hugging Face):** The Large Language Model receives the prompt and generates a "thought." This thought process might be a direct answer to you, or it might be a decision to use a tool.
4.  **The Tools (Skills):** If the LLM decides to use a tool, the AgentExecutor calls the appropriate function (`save_memory`, `find_memories`, etc.).
5.  **The Memory (Long-Term):** The tools interact directly with the `EnhancedMemoryStore`, which handles the logic of writing to the SQLite database and the FAISS vector store.
6.  **The Response:** The result of the tool's action (e.g., "✓ Memory saved") or the LLM's direct answer is passed back to the AgentExecutor, which then delivers the final response to you.

---

## 🛠️ Available Tools

The agent comes equipped with several powerful tools to assist you:

-   **save_memory**: Automatically saves important information, facts, and details to the long-term knowledge base for future reference.
-   **find_memories**: Performs semantic search across your stored memories to find relevant information based on meaning, not just keywords.
-   **set_reminder**: Creates time-based notifications that trigger with console messages, Windows desktop toasts, and audio alerts.
-   **show_reminders**: Displays your complete calendar of all reminders, including both pending and completed ones with status indicators.
-   **update_memory**: Modifies existing memories, allowing you to change content, tags, or importance levels.
-   **delete_memory**: Removes specific memories from the knowledge base.
-   **delete_reminder**: Removes one, multiple, or all reminders with flexible options.
-   **update_reminder**: Changes reminder times or content after creation.
-   **find_reminders**: Searches through reminders by content using semantic matching.
-   **show_statistics**: Provides insights into your memory usage, including total memories, reminders, and usage patterns.

---

## 🔧 Getting Started: A Step-by-Step Guide

### Prerequisites

-   **Python 3.11+**: This project requires a modern version of Python. Python 3.13 is confirmed to work.
-   **Git**: For cloning the repository.
-   **(Windows Only) C++ Build Tools**: Recommended for a smooth installation. You can get them from the [Visual Studio downloads page](https://visualstudio.microsoft.com/visual-studio-build-tools/) (select "Desktop development with C++").
-   **Platform Notes**: Windows users get additional features including desktop notifications and audio alerts for reminders. These features gracefully degrade on other platforms.

### 1. Clone the Repository

```sh
git clone <your-repository-url>
cd advanced-ai-memory-agent
```

### 2. Create and Activate a Virtual Environment

Using a virtual environment is critical to avoid conflicts with other Python projects.

```sh
# Create the virtual environment in a folder named .venv
python -m venv .venv

# Activate the environment
# On Windows (PowerShell):
.\.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```
*(Your terminal prompt should now be prefixed with `(.venv)`)*

### 3. Prepare Your `requirements.txt` File

Create a file named `requirements.txt` in the project's root directory and add the following content. This ensures you have the latest, compatible libraries.

```text
# Core AI and LangChain functionality (latest versions)
langchain
langchain-community
langchain-huggingface
langchain-core

# Hugging Face client for model inference
huggingface-hub

# Vector store for semantic search
faiss-cpu

# Sentence Transformers for embeddings
sentence-transformers

# For loading environment variables from a .env file
python-dotenv

# For rich terminal UI (colors, panels, etc.)
rich

# Pydantic is a core dependency for LangChain tool schemas
pydantic

# Windows desktop notifications and audio alerts (Windows only)
win10toast
```

### 4. Install Dependencies

With your environment active, run the following command to install all necessary packages.

```sh
pip install -r requirements.txt
```

### 5. Configure Your Hugging Face API Key

The agent needs this key to communicate with the models.

1.  Create a file named `.env` in the root directory.
2.  Add your key to this file. You can get a key from your [Hugging Face account settings](https://huggingface.co/settings/tokens).

    ```env
    HF_API_KEY="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    ```

**Note:** Hugging Face provides a limited number of free API calls. For extensive testing or production use, consider upgrading to a paid plan. The model comparison script can consume credits quickly when testing multiple models.

---

## 🚀 How to Run and Customize

### Running the Agent

Simply execute the main Python script:

```sh
python main.py
```

![Agent Interface Demo](images/Képernyőkép%202025-10-24%20204521.png)

The agent will start with a welcome message and begin the interactive conversation.

### Model Comparison Tool

The project includes a model comparison script that tests different Hugging Face models against standardized prompts to evaluate their performance as AI agents:

```sh
python compare_models.py
```

This script will:
- Test multiple models against various capability categories (Greeting & Persona, Memory Storage, Memory Retrieval, etc.)
- Display results in a formatted table
- Help you choose the best model for your needs

![Model Comparison Results](images/Képernyőkép%202025-10-24%20204521.png)

You can specify which models to test:

```sh
python compare_models.py --models "mistralai/Mistral-7B-Instruct-v0.2" "google/gemma-2b-it"
```

### Changing the LLM

The default model is `mistralai/Mistral-7B-Instruct-v0.2`. To use a different one, you can set the `MODEL` environment variable before running the script.

```sh
# Example on Windows (PowerShell)
$env:MODEL="google/gemma-2b-it"
python main.py

# Example on macOS/Linux
export MODEL="google/gemma-2b-it"
python main.py
```

---

## Future Roadmap

This agent is a strong foundation. Here are some potential directions for future development:

-   **Expanding Capabilities:**
    -   **Model Comparison Framework:** ✅ Implemented - Automated testing and comparison of different LLM models
    -   **Web Search:** Integrate a tool like Tavily or SerpAPI to allow the agent to answer questions about current events.
    -   **Calendar Integration:** Connect the `set_reminder` tool to the Google Calendar API.
    -   **File System Access:** Add tools to read, write, and summarize local files.
-   **Improving Intelligence:**
    -   **Memory Summarization:** Implement a background process that periodically summarizes older memories to keep the knowledge base concise.
    -   **Knowledge Graph:** For highly relational data, migrate from SQLite to a graph database like Neo4j to track relationships between entities.
-   **Deployment & Accessibility:**
    -   **API Endpoint:** Wrap the agent in a FastAPI server to make it accessible to other applications.
    -   **Web Frontend:** Build a user-friendly interface with Streamlit or Gradio.