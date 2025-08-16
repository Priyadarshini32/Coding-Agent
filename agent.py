"""
Core AI coding agent with Perceive -> Reason -> Act -> Learn loop.
"""
import json
import platform
import time
from terminal_interface import TerminalInterface
from llm_integration import LLMIntegration
from tools import ToolExecutionSystem
from memory_manager import MemoryManager


class Agent:
    """Core AI coding agent with Perceive -> Reason -> Act -> Learn loop."""

    def __init__(self, llm_integration, tool_execution_system, terminal_interface, project_root=None):
        """Initializes the Agent with LLM integration, tool system, terminal interface, and memory manager."""
        self.llm_integration = llm_integration
        self.tool_execution_system = tool_execution_system
        self.terminal_interface = terminal_interface
        # Use memory manager from tool execution system if available, otherwise create new one
        self.memory_manager = getattr(tool_execution_system, 'memory_manager', None)
        if not self.memory_manager:
            self.memory_manager = MemoryManager(project_root)
        self.conversation_history = []

    def perceive(self, user_input):
        """Gathers current state and adds user input to conversation history."""
        self.conversation_history.append({"role": "user", "content": user_input})
        
        # Sync memory and get current context
        self.memory_manager.sync_memory()
        current_context = self.memory_manager.get_current_context()
        
        os_info = platform.system()
        return {
            "user_input": user_input, 
            "current_context": current_context, 
            "os_info": os_info
        }

    def reason(self, perception):
        """Uses LLM to analyze situation and generate action plan."""
        # Get memory context from perception
        memory_context = perception.get("current_context", {})
        
        response_message = self.llm_integration.generate_plan(
            self.conversation_history, 
            self.tool_execution_system.tool_schemas,
            memory_context
        )
        return response_message

    def act(self, action):
        """Executes tools safely with user oversight or displays textual response."""
        tool_calls = []
        processed_action = action.strip()
        if processed_action.startswith('```json') and processed_action.endswith('```'):
            processed_action = processed_action[len('```json'):-len('```')].strip()

        try:
            response_json = json.loads(processed_action)
            if "tool_calls" in response_json:
                tool_calls = response_json["tool_calls"]
            elif "text" in response_json:
                agent_response_content = response_json['text']
                self.terminal_interface.display_message(f"Agent response: {agent_response_content}")
                self.conversation_history.append({"role": "model", "content": agent_response_content})
                return {"status": "success", "message": agent_response_content}

        except json.JSONDecodeError:
            self.terminal_interface.display_message(f"Agent response: {action}")
            self.conversation_history.append({"role": "model", "content": action})
            return {"status": "success", "message": action}

        if tool_calls:
            for tool_call in tool_calls:
                self.terminal_interface.display_tool_call(tool_call)
                self.conversation_history.append({"role": "model", "content": f"TOOL_CALL: {json.dumps(tool_call)}"})

                function_name = tool_call["function"]["name"]
                tool_args = tool_call["function"]["arguments"]
                
                # Handle approval for destructive actions
                if function_name in ["write_file", "delete_file", "clear_file_content", "apply_code_change"]:
                    action_description = f"The agent wants to execute '{function_name}' on '{tool_args.get('filepath', '')}'."
                    preview_content = None
                    language = None

                    if function_name == "write_file":
                        preview_content = tool_args.get('content', '')
                        language = "text" # Can be improved to detect language based on filepath
                    elif function_name == "apply_code_change":
                        # For apply_code_change, show a simplified diff or just old vs new
                        old_code = tool_args.get('old_code', '')
                        new_code = tool_args.get('new_code', '')
                        preview_content = f"--- Old Code ---\n{old_code}\n+++ New Code +++\n{new_code}"
                        language = "diff" # Use diff language for highlighting

                    if not self.terminal_interface.confirm_action(action_description, preview_content, language):
                        self.terminal_interface.display_message("Action cancelled by user.", style="red")
                        self.conversation_history.append({"role": "user_action", "content": "User denied the action."})
                        return {"status": "cancelled", "message": "Action cancelled by user."}

                if function_name in self.tool_execution_system.available_tools:
                    # Record tool execution start time
                    start_time = time.time()
                    
                    tool_output = self.tool_execution_system.execute_tool_from_dict(
                        {"function": {"name": function_name, "arguments": tool_args}}
                    )
                    
                    # Calculate execution time
                    execution_time = time.time() - start_time
                    
                    # Add execution time to tool output
                    tool_output["execution_time"] = execution_time
                    tool_output["tool_name"] = function_name
                    
                    # Record tool usage in memory
                    success = tool_output.get("status") == "success"
                    error_message = tool_output.get("message") if not success else None
                    
                    self.memory_manager.record_tool_usage(
                        function_name, success, execution_time, error_message,
                        context={"arguments": tool_args}
                    )
                    
                    # Record file operations if applicable
                    if function_name in ["read_file", "write_file", "delete_file", "clear_file_content", "apply_code_change"]:
                        filepath = tool_args.get("filepath")
                        if filepath:
                            operation = function_name
                            success = tool_output.get("status") == "success"
                            error_message = tool_output.get("message") if not success else None
                            
                            self.memory_manager.record_file_operation(
                                filepath, operation, success, error_message=error_message
                            )
                    
                    # Cache file content for read operations
                    if function_name == "read_file" and tool_output.get("status") == "success":
                        filepath = tool_args.get("filepath")
                        content = tool_output.get("content", "")
                        if filepath and content:
                            self.memory_manager.cache_file_content(filepath, content, "read")
                    
                    self.terminal_interface.display_tool_output(tool_output)
                    self.conversation_history.append({"role": "tool_output", "content": json.dumps(tool_output)})
                    return tool_output
                else:
                    error_message = f"Tool {function_name} not found."
                    self.terminal_interface.display_message(error_message, style="red")
                    self.conversation_history.append({"role": "tool_output", "content": json.dumps(
                        {"status": "error", "message": error_message}
                    )})
                    return {"status": "error", "message": error_message}
        return {"status": "error", "message": "No tool calls detected or text response from LLM."} # Ensure a return for all paths


    def learn(self, observation):
        """Updates memory and context for future decisions."""
        # Extract learning patterns from the observation
        if isinstance(observation, dict):
            # Record tool usage if it's a tool execution result
            if "status" in observation:
                tool_name = observation.get("tool_name", "unknown")
                success = observation.get("status") == "success"
                execution_time = observation.get("execution_time")
                error_message = observation.get("message") if not success else None
                
                self.memory_manager.record_tool_usage(
                    tool_name, success, execution_time, error_message
                )
            
            # Record file operations
            if "filepath" in observation:
                filepath = observation["filepath"]
                operation = observation.get("operation", "unknown")
                success = observation.get("status") == "success"
                error_message = observation.get("message") if not success else None
                
                self.memory_manager.record_file_operation(
                    filepath, operation, success, error_message=error_message
                )
        
        # Learn from session periodically
        if len(self.conversation_history) % 10 == 0:  # Every 10 interactions
            self.memory_manager.learn_from_session()

    def run(self, user_input):
        self.perceive(user_input)
        
        MAX_FIX_ATTEMPTS = 3 # Limit to prevent infinite loops
        attempts = 0
        
        current_user_input = user_input

        while attempts < MAX_FIX_ATTEMPTS:
            action = self.reason(current_user_input)

            if isinstance(action, str) and action.strip().startswith('```json'):
                observation = self.act(action)
                
                if observation and observation.get("status") == "success" and "content" in observation and "pytest" in observation.get("content", '') and "failed" in observation.get("content", ''):
                    # If run_tests tool was just executed and tests failed
                    self.terminal_interface.display_message("Tests failed. Attempting to fix...", style="red")
                    feedback_request = f"The tests failed with the following output: {observation['content']}. Please analyze this output and provide the exact `apply_code_change` tool call to fix the issue. If you cannot fix it, explain why." 
                    
                    # Set current_user_input to the feedback request to guide next LLM call
                    current_user_input = feedback_request
                    attempts += 1
                    continue # Loop back to reason for a fix
                elif observation and observation.get("status") == "success" and "content" in observation and "pytest" in observation.get("content", '') and "passed" in observation.get("content", ''):
                    # If run_tests tool was just executed and tests passed
                    self.terminal_interface.display_message("Tests passed successfully!", style="green")
                    self.conversation_history.append({"role": "model", "content": "Tests passed successfully!"})
                    return observation # Exit loop if tests pass
                
                # If a tool was executed but not run_tests, or it's an error
                if observation and observation.get("status") in ["success", "error"]:
                    feedback_request = (
                        f"Based on the tool execution output: {json.dumps(observation)}, "
                        "provide a concise summary or feedback."
                    )
                    feedback_response = self.llm_integration.generate_plan(
                        self.conversation_history + [{"role": "user", "content": feedback_request}],
                        self.tool_execution_system.tool_schemas
                    )
                    try:
                        feedback_json = json.loads(feedback_response.strip().replace('```json', '').replace('```', ''))
                        if "text" in feedback_json:
                            self.terminal_interface.display_message(
                                feedback_json['text'], style="yellow", title="Agent Feedback"
                            )
                            self.conversation_history.append({"role": "model", "content": feedback_json['text']})
                        else:
                            self.terminal_interface.display_message(
                                feedback_response, style="yellow", title="Agent Feedback"
                            )
                            self.conversation_history.append({"role": "model", "content": feedback_response})
                    except json.JSONDecodeError:
                        self.terminal_interface.display_message(
                            feedback_response, style="yellow", title="Agent Feedback"
                        )
                        self.conversation_history.append({"role": "model", "content": feedback_response})

                    return observation # Return the actual tool output
                return observation
            else:
                # If LLM didn't suggest a tool call, it's a direct textual response
                self.terminal_interface.display_message(action, title="Agent Response")
                self.conversation_history.append({"role": "model", "content": action})
                return {"status": "success", "message": action}
        
        self.terminal_interface.display_message("Max fix attempts reached. Could not resolve the issue automatically.", style="red")
        return {"status": "error", "message": "Max fix attempts reached."}
