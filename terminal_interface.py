from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.syntax import Syntax
from rich.markdown import Markdown # Import Markdown
import json

class TerminalInterface:
    def __init__(self):
        self.console = Console()

    def display_message(self, message, style="green", title=None):
        # Check if the message contains a markdown code block
        if "```" in message:
            parts = message.split("```")
            for i, part in enumerate(parts):
                if i % 2 == 1: # This is a code block
                    code_lines = part.split('\n')
                    lang = 'text' # Default language to text if not specified
                    if code_lines and code_lines[0].strip() in ['python', 'json', 'bash']:
                        lang = code_lines[0].strip()
                        code = '\n'.join(code_lines[1:]).strip()
                    else:
                        code = part.strip() # No language specified, take whole part

                    syntax = Syntax(code, lang, theme="monokai", line_numbers=False)
                    self.console.print(Panel(syntax, title=title if title else "Code"))
                else: # This is regular text
                    if part.strip(): # Only print if there's actual text content
                        # Use rich.markdown.Markdown for regular text parts
                        self.console.print(Panel(Markdown(part.strip()), title=title))
        else:
            # If no code block, treat the whole message as Markdown
            self.console.print(Panel(Markdown(message), title=title))

    def get_user_input(self, prompt="You: "):
        return self.console.input(f"[bold blue]{prompt}[/bold blue]")

    def display_agent_thought(self, thought):
        # Make agent thought more subtle, as display_tool_call will be prominent
        self.console.print(Text(f"[italic grey]Agent thinking: {thought}[/italic grey]"))

    def display_tool_call(self, tool_call_json):
        self.console.print(Panel(Text(f"Agent Decided to Call Tool:\n{json.dumps(tool_call_json, indent=2)}", style="magenta"), title="Agent Action"))

    def display_tool_output(self, output):
        if isinstance(output, dict) and output.get("status") == "success" and "content" in output:
            self.console.print(Panel(Text(f"Tool Execution Successful!\nContent:\n{output['content']}", style="green"), title="Tool Output"))
        elif isinstance(output, dict) and output.get("status") == "error":
            self.console.print(Text(f"[bold red]Error: {output['message']}[/bold red]")) # Simpler error message
        else:
            self.console.print(Panel(Text(f"Tool Output: {output}", style="cyan"), title="Tool Output"))

    def confirm_action(self, action_description, preview_content=None, language=None):
        self.console.print(Panel(Text(action_description, style="bold yellow"), title="Action Confirmation"))

        if preview_content:
            self.console.print(Panel(Text("Preview of change:", style="italic blue"), title="Preview"))
            if language:
                syntax = Syntax(preview_content, language, theme="monokai", line_numbers=True)
                self.console.print(Panel(syntax, title="Code Preview"))
            else:
                self.console.print(Panel(Text(preview_content), title="Content Preview"))
        
        response = self.console.input("[bold magenta]Do you approve this action? (yes/no)[/bold magenta] ").lower()
        return response == 'yes'
