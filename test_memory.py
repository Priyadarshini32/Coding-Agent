"""
Test script for the memory management system
"""
import os
import sys
from memory_manager import MemoryManager
from working_memory import WorkingMemory
from persistent_memory import PersistentMemory


def test_memory_system():
    """Test the memory management system."""
    print("Testing Memory Management System...")
    
    # Create memory manager
    memory_manager = MemoryManager()
    
    # Test file content caching
    print("\n1. Testing file content caching...")
    test_content = "print('Hello, World!')"
    memory_manager.cache_file_content("test_file.py", test_content, "write")
    
    cached_content = memory_manager.get_file_content("test_file.py")
    print(f"Cached content: {cached_content}")
    
    # Test file operation recording
    print("\n2. Testing file operation recording...")
    memory_manager.record_file_operation("test_file.py", "write", True, {"lines_added": 1})
    memory_manager.record_file_operation("test_file.py", "read", True)
    
    # Test tool usage recording
    print("\n3. Testing tool usage recording...")
    memory_manager.record_tool_usage("read_file", True, 0.1, context={"filepath": "test_file.py"})
    memory_manager.record_tool_usage("write_file", True, 0.2, context={"filepath": "test_file.py"})
    memory_manager.record_tool_usage("run_command", False, 1.0, "Command not found", context={"command": "invalid_cmd"})
    
    # Test pattern recording
    print("\n4. Testing pattern recording...")
    memory_manager.record_project_pattern("file_modification", {
        "file_type": ".py",
        "operation": "write",
        "lines_changed": 1
    }, filepath="test_file.py")
    
    memory_manager.record_user_preference("coding_style", {
        "indentation": "spaces",
        "line_length": 80,
        "preferred_language": "python"
    })
    
    memory_manager.record_success_pattern("test_fix", {
        "error_type": "syntax_error",
        "fix_method": "add_missing_colon"
    }, 0.9, context={"file_type": ".py"})
    
    # Test code snippet storage
    print("\n5. Testing code snippet storage...")
    memory_manager.store_code_snippet(
        "def hello_world():\n    print('Hello, World!')",
        "function_definition",
        context={"language": "python"},
        tags=["hello", "function"],
        filepath="test_file.py"
    )
    
    # Test context retrieval
    print("\n6. Testing context retrieval...")
    context = memory_manager.get_current_context()
    print(f"Current context keys: {list(context.keys())}")
    
    # Test pattern retrieval
    print("\n7. Testing pattern retrieval...")
    patterns = memory_manager.get_relevant_patterns({"file_type": ".py"})
    print(f"Found {len(patterns)} relevant patterns")
    
    # Test memory summary
    print("\n8. Testing memory summary...")
    summary = memory_manager.get_memory_summary()
    print(f"Memory summary: {summary}")
    
    # Test code snippet search
    print("\n9. Testing code snippet search...")
    snippets = memory_manager.search_code_snippets(query="hello", snippet_type="function_definition")
    print(f"Found {len(snippets)} code snippets")
    
    # Test frequently accessed files
    print("\n10. Testing frequently accessed files...")
    frequent_files = memory_manager.get_frequently_accessed_files()
    print(f"Frequently accessed files: {frequent_files}")
    
    print("\nMemory system test completed successfully!")


def test_working_memory():
    """Test the working memory system."""
    print("\nTesting Working Memory System...")
    
    working_memory = WorkingMemory()
    
    # Test file content caching
    working_memory.cache_file_content("test_working.py", "def test():\n    pass")
    
    # Test change tracking
    working_memory.record_file_operation("test_working.py", "write", True)
    working_memory.record_command("python test_working.py", True, "No output")
    
    # Test session summary
    summary = working_memory.get_session_summary()
    print(f"Working memory summary: {summary}")
    
    print("Working memory test completed!")


def test_persistent_memory():
    """Test the persistent memory system."""
    print("\nTesting Persistent Memory System...")
    
    persistent_memory = PersistentMemory()
    
    # Test file access recording
    persistent_memory.record_file_access("test_persistent.py", "read", True, "abc123", 100)
    
    # Test tool effectiveness
    persistent_memory.record_tool_usage("read_file", True, 0.1)
    
    # Test pattern recording
    persistent_memory.record_project_pattern("test_pattern", {"type": "test"}, filepath="test_persistent.py")
    
    # Test memory summary
    summary = persistent_memory.get_memory_summary()
    print(f"Persistent memory summary: {summary}")
    
    print("Persistent memory test completed!")


if __name__ == "__main__":
    try:
        test_memory_system()
        test_working_memory()
        test_persistent_memory()
        print("\nAll memory tests passed!")
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
