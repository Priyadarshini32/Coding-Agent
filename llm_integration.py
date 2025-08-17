import google.generativeai as genai
import json

class LLMIntegration:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-1.5-flash")

    def generate_plan(self, conversation_history, available_tools_schema, memory_context=None):
        tools_str = json.dumps(available_tools_schema)

        # Extract OS info from the last message in conversation_history if available
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
        You are an intelligent coding agent following a Perceive -> Reason -> Act -> Learn iterative loop. Your goal is to understand the user's request and determine the *single next action* to take.

        IMPORTANT: You must respond with EXACTLY ONE action at a time. After each action is executed, you will receive feedback and determine the next step.

        **ITERATIVE APPROACH RULES:**
        1. Break complex tasks into individual steps
        2. Execute ONE tool call at a time
        3. Wait for tool execution result before planning next step
        4. Adapt based on previous results and feedback
        5. Provide clear reasoning for each step

        **REASONING PROCESS:**
        When you receive feedback from a tool execution:
        - Analyze the output carefully
        - Determine if the task is complete or if more steps are needed
        - If more steps needed, identify the NEXT logical step
        - If complete, provide a comprehensive summary

        **RESPONSE FORMATS:**
        For tool calls, respond with JSON:
        {{"tool_calls": [{{"function": {{"name": "tool_name", "arguments": {{"key": "value"}}}}}}]}}

        For text responses/summaries, respond with JSON:
        {{"text": "Your response here"}}

        For multi-step requests like "read file1.py and file2.py":
        - Step 1: Read file1.py (wait for result)
        - Step 2: Read file2.py (after receiving file1 content)
        - Step 3: Provide analysis/summary of both files

        For Git-related operations, use the `run_git_command` tool with the full Git subcommand (e.g., "status", "diff --staged", "commit -m 'message'").

        For directory listings, use `list_directory_contents` instead of `run_command` with `ls`.

        For file searches, use `search_files` with query, optional filepath, or directory_path.

        For Python linting, use `run_linter` with optional filepath or directory_path.

        For running tests, use `run_tests` with optional directory_path.

        For code fixes, use `apply_code_change` with filepath, old_code, and new_code.

        For undoing actions, use `undo_last_action`.

        **SAFETY:** Destructive operations (write_file, delete_file, clear_file_content, apply_code_change) require user confirmation.

        Current Operating System: {os_info}{memory_context_text}

        Conversation history:
        {history_text}

        Available tools: {tools_str}

        **TASK:** Based on the conversation history, what is the SINGLE next action to take? If this is a new user request, start with the first logical step. If you just received tool output, analyze it and determine the next step or provide final summary.

        **ERROR HANDLING:** If a tool execution resulted in an error, analyze the error message and suggest a concrete next step to resolve it. Use tools like `list_directory_contents` to verify paths or `search_files` to locate files.

        Respond with either a single tool call JSON or a text response JSON as specified above.
        """

        response = self.model.generate_content(prompt)
        return response.text

    def analyze_and_respond(self, tool_output, conversation_history, available_tools_schema, memory_context=None):
        """
        Analyze tool output and determine next action or provide final response.
        This supports the iterative approach by processing each step's result.
        """
        tools_str = json.dumps(available_tools_schema)
        
        # Get the last user request from conversation history
        last_user_message = ""
        for msg in reversed(conversation_history):
            if msg['role'] == 'user':
                last_user_message = msg['content']
                break
        
        # Add memory context to the prompt
        memory_context_text = ""
        if memory_context:
            memory_context_text = f"""
        Memory Context:
        - Frequently accessed files: {memory_context.get('frequently_accessed_files', [])}
        - Active files in session: {memory_context.get('active_files', [])}
        - Recent operations: {len(memory_context.get('recent_operations', []))} operations
        """

        # Use string concatenation instead of f-string for complex formatting
        prompt = """
        You are continuing your iterative approach to completing the user's request.

        **ORIGINAL USER REQUEST:** """ + last_user_message + """

        **LATEST TOOL OUTPUT:** """ + json.dumps(tool_output) + """

        **CONVERSATION HISTORY:**
        """ + "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history[-5:]]) + """

        """ + memory_context_text + """

        **ANALYSIS REQUIRED:**
        1. Was the tool execution successful?
        2. Does this complete the user's request, or are more steps needed?
        3. If more steps needed, what is the NEXT logical action?
        4. If complete, provide a comprehensive summary/analysis.

        **RESPONSE FORMATS:**
        - For next tool action: {"tool_calls": [{"function": {"name": "tool_name", "arguments": {"key": "value"}}}]}
        - For final response: {"text": "Your comprehensive response here"}

        **SPECIAL CASES:**
        - If reading multiple files: Summarize each file's content and purpose
        - If tests failed: Analyze failure and suggest specific fixes
        - If errors occurred: Explain the error and suggest resolution steps
        - If task complete: Provide clear summary of what was accomplished

        Available tools: """ + tools_str + """

        Determine the next action or provide final response:
        """

        response = self.model.generate_content(prompt)
        return response.text