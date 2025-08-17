import google.generativeai as genai
import json

class LLMIntegration:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-1.5-flash")

    def generate_response_feedback(self, user_request, agent_response, tool_output=None):
        """Generate feedback on the agent's response quality and effectiveness."""
        feedback_prompt = f"""
        You are evaluating an AI coding agent's response to assess its quality and effectiveness.
        
        **USER REQUEST:** {user_request}
        
        **AGENT RESPONSE:** {agent_response}
        
        **TOOL OUTPUT (if any):** {json.dumps(tool_output) if tool_output else "No tool execution"}
        
        Provide constructive feedback on the agent's response considering:
        1. **Accuracy**: Was the response factually correct and relevant?
        2. **Completeness**: Did it fully address the user's request?
        3. **Clarity**: Was the explanation clear and well-structured?
        4. **Usefulness**: How helpful was the response for the user?
        5. **Efficiency**: Was the approach taken optimal?
        
        Provide a brief evaluation (2-3 sentences) highlighting strengths and areas for improvement.
        Rate the response: Excellent/Good/Fair/Poor
        
        Format: "**Agent Feedback:** [Your evaluation] **Rating:** [Rating]"
        """
        
        try:
            feedback_response = self.model.generate_content(feedback_prompt)
            return feedback_response.text.strip()
        except Exception as e:
            return f"**Agent Feedback:** Unable to generate feedback due to error: {str(e)} **Rating:** N/A"

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

        **RESPONSE FORMATS:**
        For tool calls, respond with JSON:
        {{"tool_calls": [{{"function": {{"name": "tool_name", "arguments": {{"key": "value"}}}}}}]}}

        For text responses/summaries, respond with JSON:
        {{"text": "Your response here"}}

        For multi-step requests like "read file1.py and file2.py":
        - Step 1: Read file1.py (wait for result)
        - Step 2: Read file2.py (after receiving file1 content)
        - Step 3: Provide analysis/summary of both files

        For Git-related operations, use the `run_git_command` tool with the full Git subcommand (e.g., "status", "diff", "commit -m 'message'").

        For directory listings, use `list_directory_contents` instead of `run_command` with `ls`.

        For file searches, use `search_files` with query, optional filepath, or directory_path.

        For Python linting, use `run_linter` with optional filepath or directory_path.

        For running tests, use `run_tests` with optional directory_path.

        For code fixes, use `apply_code_change` with filepath, old_code, and new_code.

        For undoing actions, explicitly use the `undo_last_action` tool. If a user asks to undo, the next step should always be to call `undo_last_action`.

        **SAFETY:** Destructive operations (write_file, delete_file, clear_file_content, apply_code_change) require user confirmation.

        Current Operating System: {os_info}{memory_context_text}

        Conversation history:
        {history_text}

        Available tools: {tools_str}

        **TASK:** Based on the conversation history, what is the SINGLE next action to take?
        1. If the user's request is a general knowledge question about a specific file (e.g., "what is package.json", "what is requirements.txt"), first list directory contents to show available files, then provide a comprehensive explanation.
        2. If the user's request is general knowledge without file context (e.g., "what is Python", "explain OOP"), provide a comprehensive text response directly.
        3. Always use `list_directory_contents` first when users ask about specific file types to show what files are actually available.
        4. If you just received tool output and the task is complete, provide a final summary.
        
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
        
        # Enhanced handling - proactive approach for "what is" questions
        # Since we're now using list_directory_contents first, we don't need special file not found handling

        # Handle successful directory listing for "what is" questions
        if (tool_output.get('status') == 'success' and 
            tool_output.get('tool_name') == 'list_directory_contents' and
            any(phrase in last_user_message.lower() for phrase in ["what is", "what's", "explain"])):
            
            # Get the file type from the original user question
            file_type_mentioned = "unknown file"
            user_question_words = last_user_message.lower().split()
            for i, word in enumerate(user_question_words):
                if word in ["what", "what's"] and i + 1 < len(user_question_words) and user_question_words[i + 1] == "is":
                    if i + 2 < len(user_question_words):
                        file_type_mentioned = user_question_words[i + 2]
                        break
            
            directory_contents = tool_output.get('content', 'No files found')
            
            # Generate detailed explanation using LLM
            explanation_prompt = f"""
            A user asked "what is {file_type_mentioned}" and we've listed the current directory contents.
            
            Current directory contents:
            {directory_contents}
            
            The file "{file_type_mentioned}" is not present in this directory. Provide a comprehensive explanation that includes:
            1. A clear statement that the file was not found in the current directory
            2. Show the current directory contents in a readable format
            3. What the file type "{file_type_mentioned}" is and its typical purpose
            4. Why it might not be present in this project (analyze the directory contents to understand the project type)
            5. A realistic example of what this file typically looks like with proper code formatting
            6. Related alternatives that might be used instead based on the project type
            
            Format your response with proper markdown including code blocks where appropriate.
            Make it detailed and educational - much more than a brief explanation.
            Be specific about why this file type might not be relevant to the current project based on the visible files.
            """
            
            explanation_response = self.model.generate_content(explanation_prompt)
            
            final_response = f"""{explanation_response.text}"""
            
            return json.dumps({"text": final_response})

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
        - If the previous tool call was `undo_last_action` and it was successful, the task is complete.
        - If a file search or read for a file returns no results, provide a final summary explaining what the file is and why it might not be present, then list the contents of the current directory to be helpful.
        - If you have successfully listed the directory contents in a previous step to generate a `README.md`, the next logical step is to read the contents of all the relevant project files in the directory to gather information.

        Available tools: """ + tools_str + """

        Determine the next action or provide final response:
        """

        response = self.model.generate_content(prompt)
        return response.text