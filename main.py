import os
from dotenv import load_dotenv
from agent import Agent
from llm_integration import LLMIntegration
from tools import ToolExecutionSystem
from terminal_interface import TerminalInterface
from action_history import ActionHistory # Import ActionHistory

def main():
    load_dotenv()
    google_api_key = os.getenv("GOOGLE_API_KEY")

    if not google_api_key:
        print("Error: GOOGLE_API_KEY environment variable not set.")
        return

    llm_integration = LLMIntegration(api_key=google_api_key)
    terminal_interface = TerminalInterface()
    action_history = ActionHistory() # Initialize ActionHistory
    
    # Get current project root for memory management
    project_root = os.getcwd()
    
    # Create memory manager first
    from memory_manager import MemoryManager
    memory_manager = MemoryManager(project_root)
    
    # Pass memory manager to tool execution system
    tool_execution_system = ToolExecutionSystem(action_history, memory_manager)
    
    agent = Agent(llm_integration, tool_execution_system, terminal_interface, project_root)

    terminal_interface.display_message("Welcome to the AI Coding Agent! Type 'exit' to quit.")
    terminal_interface.display_message("""I can:
- Read, write, delete, and clear content of files.
- Search for text within files.
- List directory contents.
- Perform various Git operations (e.g., status, diff, commit, branch, log).
- Run arbitrary shell commands.
- Perform code analysis (linting, testing), and attempt to automatically fix failing tests.
- Answer general coding and project-related questions.""", title="What I can do:")

    while True:
        user_input = terminal_interface.get_user_input()
        if user_input.lower() == 'exit':
            terminal_interface.display_message("Exiting agent. Goodbye!")
            break

        agent.run(user_input)
        # terminal_interface.display_message(f"Observation: {observation}") # Removed raw observation display

if __name__ == "__main__":
    main()
