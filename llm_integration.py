import google.generativeai as genai
import json

class LLMIntegration:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-1.5-flash")

    def generate_plan(self, conversation_history, available_tools_schema, memory_context=None):
        tools_str = json.dumps(available_tools_schema)

        # Extract OS info from the last message in conversation_history if available
        # Assuming the perception object with os_info is added as part of the conversation history
        os_info = "Unknown"
        if conversation_history and isinstance(conversation_history[-1], dict) and "os_info" in conversation_history[-1]:
            os_info = conversation_history[-1]["os_info"]

        history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history])
        
        # Add memory context to the prompt
        memory_context_text = ""
        if memory_context:
            memory_context_text = f"""
        Memory Context:
        - Frequently accessed files: {memory_context.get('frequently_accessed_files', [])}
        - Active files in session: {memory_context.get('active_files', [])}
        - Recent operations: {len(memory_context.get('recent_operations', []))} operations
        - Tool effectiveness: {list(memory_context.get('tool_effectiveness', {}).keys())}
        - User preferences: {list(memory_context.get('user_preferences', {}).keys())}
        """

        prompt = f"""
        You are a coding agent. Your goal is to understand the user's request, analyze the current state, and decide on the next action to take using the available tools. After performing an action, provide a summary or feedback to the user. If the user asks a definitional or informational question (e.g., "What is X?"), provide a comprehensive explanation, including relevant examples formatted as code blocks where appropriate.

        For any Git-related operations (e.g., status, diff, commit, branch, pull, push), use the `run_git_command` tool. The `command` argument for `run_git_command` should be the full Git subcommand and its arguments as a single string (e.g., "status", "diff --staged", "commit -m 'Initial commit'", "pull origin main"). These commands are executed directly by Git through the shell.

        For listing directory contents (e.g., "what is in my current directory", "list files"), use the `list_directory_contents` tool. Always prefer `list_directory_contents` over `run_command` with `ls` or `dir` for listing files or directories, as it is cross-platform and more reliable.

        For searching content within files (e.g., "find 'hello' in main.py", "search for 'TODO' in the project"), use the `search_files` tool. Specify the `query` argument (the text to search for). You can optionally provide `filepath` to search in a specific file or `directory_path` to search in a specific folder. If neither is provided, the current directory will be searched.

        For linting Python code (e.g., "lint this file", "check code quality"), use the `run_linter` tool. You can provide an optional `filepath` or `directory_path`.

        For running Python tests (e.g., "run tests", "execute all tests"), use the `run_tests` tool. You can provide an optional `directory_path`.

        To fix code, you can use the `apply_code_change` tool. This tool requires `filepath`, `old_code` (the exact string to be replaced), and `new_code` (the replacement string). Use this tool to fix bugs or refactor code based on analysis or test results.

        To undo the last destructive action (e.g., file write, delete, clear, or code change), use the `undo_last_action` tool. This tool reverses the most recent change recorded in the agent's action history. Use it cautiously, as it can only undo the very last action.

        For executing any other general shell commands (e.g., `pip install`, `npm test`, `python script.py`), use the `run_command` tool. The `command` argument should be the full shell command string (e.g., "echo Hello", "python my_script.py").

        **Important Safety Note:** For destructive operations like `write_file`, `delete_file`, `clear_file_content`, and `apply_code_change`, the agent will always ask for user confirmation with a preview of the changes. Be prepared to approve or deny these actions.

        Current Operating System: {os_info}{memory_context_text}

        Conversation history:
        {history_text}

        Available tools: {tools_str}

        Based on the conversation and available tools, what is the next action? If you are reading a file, provide a summary or feedback about its content after the read. Respond with a JSON object representing the tool call, like this:
        {{"tool_calls": [{{\"function\": {{\"name\": \"tool_name\", \"arguments\": {{}}\}}}}]}}
        If no tool is suitable or if you are providing feedback, provide a textual response. When providing a textual response or summary, format it clearly, using Markdown for code examples and structured information.

        **Error Handling:** If a tool execution results in an error (i.e., the `tool_output` has `"status": "error"` and a `"message"` field), carefully analyze the error message. Explain the error to the user in clear, concise terms. If possible, suggest concrete steps or a specific tool call to resolve the issue. For example, if a file is not found, suggest using `list_directory_contents` to verify the path or `search_files` to locate it. Always prioritize providing actionable advice for error scenarios.
        """

        response = self.model.generate_content(prompt)
        return response.text
