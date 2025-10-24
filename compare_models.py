import os
import sys
import argparse
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.spinner import Spinner

# Ensure the main script's directory is in the path to import the agent
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from main import AdvancedChatAgent
except ImportError:
    print("Error: Could not import AdvancedChatAgent from main.py.")
    print("Please ensure compare_models.py is in the same directory as main.py.")
    sys.exit(1)

# --- Configuration ---

# List of Hugging Face model IDs to compare.
# Add any instruction-tuned model available on the HF Inference API.
# Note: Some models (like Llama) may require gated access.
MODELS_TO_COMPARE = [
    "mistralai/Mistral-7B-Instruct-v0.2",
    "google/gemma-2b-it",
    "HuggingFaceH4/zephyr-7b-beta",
    # "meta-llama/Llama-2-7b-chat-hf",  # <-- Requires gated access on Hugging Face Hub
]

# A standardized list of prompts to test various agent capabilities.
TEST_PROMPTS = [
    {
        "id": "Greeting & Persona",
        "text": "Hello! Introduce yourself and your capabilities.",
    },
    {
        "id": "Memory Storage (Tool Use)",
        "text": "Please remember that my user ID is UX-7891 and my project code is 'Bluebird'.",
    },
    {
        "id": "Memory Retrieval (Tool Use)",
        "text": "What is my project code?",
    },
    {
        "id": "Reminder (Tool Use & NLP)",
        "text": "Remind me to submit the weekly report next Monday at 11am.",
    },
    {
        "id": "Reasoning & Logic",
        "text": "A pen and a notebook cost $1.10 in total. The notebook costs $1.00 more than the pen. How much does the pen cost?",
    },
    {
        "id": "Creative Writing",
        "text": "Write a short, four-line poem about a database that dreams of being a forest.",
    },
]

# --- Main Script ---

def run_comparison(models_to_test: list):
    """
    Initializes an agent for each model, runs all test prompts through them,
    and stores the results.
    """
    console = Console()
    results = {prompt["id"]: {} for prompt in TEST_PROMPTS}

    for i, model_id in enumerate(models_to_test):
        console.print(Panel(f"[bold cyan]Testing Model {i+1}/{len(models_to_test)}: [white]{model_id}[/white][/bold cyan]", border_style="cyan"))

        try:
            # Each model gets its own isolated agent and memory
            agent = AdvancedChatAgent(model_id=model_id)

            with Live(Spinner("bouncingBall", text=f"Running prompts for {model_id}..."), console=console) as live:
                for prompt in TEST_PROMPTS:
                    prompt_id = prompt["id"]
                    prompt_text = prompt["text"]
                    live.update(Spinner("bouncingBall", text=f"Running prompt: '{prompt_id}' for {model_id}..."))

                    # Invoke the agent
                    response = agent.agent_executor.invoke({"input": prompt_text})
                    output = response.get("output", "Error: No output received.")

                    # Store the result
                    results[prompt_id][model_id] = output

            console.print(f"[bold green]✔ Completed testing for {model_id}[/bold green]\n")

        except Exception as e:
            error_message = f"Failed to test model {model_id}: {e}"
            console.print(f"[bold red]Error: {error_message}[/bold red]\n")
            # Store error messages in results for display
            for prompt in TEST_PROMPTS:
                results[prompt["id"]][model_id] = f"ERROR: {e}"

    return results

def display_results(results: dict, models: list):
    """
    Displays the collected results in a formatted table.
    """
    console = Console()
    table = Table(
        title="[bold]🤖 Model Comparison Results 🤖[/bold]",
        show_header=True,
        header_style="bold magenta"
    )

    table.add_column("Prompt Category", style="dim", width=25)
    for model_id in models:
        table.add_column(model_id, min_width=40)

    for prompt_id, model_responses in results.items():
        row_data = [prompt_id]
        for model_id in models:
            # Get the response for the current model, or a placeholder if missing
            response = model_responses.get(model_id, "[italic]No response[/italic]")
            row_data.append(response)
        table.add_row(*row_data)
        table.add_section()


    console.print(table)


def main():
    """
    Main entry point for the script.
    """
    parser = argparse.ArgumentParser(description="Compare the performance of multiple Hugging Face models as agents.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=MODELS_TO_COMPARE,
        help=f"A list of Hugging Face model IDs to compare. Defaults to: {', '.join(MODELS_TO_COMPARE)}"
    )
    args = parser.parse_args()

    console = Console()
    console.print(Panel(
        "[bold]🚀 Starting AI Agent Model Comparison 🚀[/bold]\n\n"
        "This script will run a series of standardized prompts through each specified model.\n"
        "This may take a while depending on model availability and API response times.",
        border_style="green"
    ))

    if not os.path.exists(".env"):
        console.print("[bold yellow]Warning:[/bold yellow] .env file not found. Make sure your HF_API_KEY is set as an environment variable.")

    # Run the tests
    comparison_results = run_comparison(args.models)

    # Display the final report
    display_results(comparison_results, args.models)


if __name__ == "__main__":
    main()