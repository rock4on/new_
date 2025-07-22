#!/usr/bin/env python3
"""
Interactive Chat Interface for the Lease Document Processing Agent
Provides a conversational interface to interact with the agent
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import readline  # For better command line experience

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from lease_agent import LeaseDocumentAgent
from config import get_config


class ChatHistory:
    """Manages chat history persistence"""
    
    def __init__(self, history_file: str, max_history: int = 100):
        self.history_file = Path(history_file)
        self.max_history = max_history
        self.messages: List[Dict[str, Any]] = []
        self.load_history()
    
    def load_history(self):
        """Load chat history from file"""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r') as f:
                    data = json.load(f)
                    self.messages = data.get('messages', [])
                    # Limit to max history
                    if len(self.messages) > self.max_history:
                        self.messages = self.messages[-self.max_history:]
            except Exception as e:
                print(f"⚠️  Could not load chat history: {e}")
                self.messages = []
    
    def save_history(self):
        """Save chat history to file"""
        try:
            # Ensure directory exists
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                'messages': self.messages[-self.max_history:],
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.history_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"⚠️  Could not save chat history: {e}")
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Add message to history"""
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat()
        }
        if metadata:
            message['metadata'] = metadata
        
        self.messages.append(message)
        
        # Auto-save every 5 messages
        if len(self.messages) % 5 == 0:
            self.save_history()
    
    def get_recent_messages(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent messages"""
        return self.messages[-count:]
    
    def clear_history(self):
        """Clear chat history"""
        self.messages = []
        if self.history_file.exists():
            self.history_file.unlink()


class LeaseAgentChat:
    """Interactive chat interface for the Lease Document Agent"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config = get_config(config_file)
        self.agent = None
        self.chat_history = None
        self.session_start = datetime.now()
        
        # Initialize components
        self._initialize_agent()
        self._initialize_chat_history()
    
    def _initialize_agent(self):
        """Initialize the lease document agent"""
        try:
            # Validate configuration
            is_valid, missing_fields = self.config.validate()
            if not is_valid:
                print("❌ Configuration is incomplete!")
                print("Missing required fields:")
                for field in missing_fields:
                    print(f"   - {field}")
                print(f"\n💡 Run 'python config.py' to create a sample configuration")
                sys.exit(1)
            
            print("\n🔧 Configuration Details:")
            print(f"   Azure Form Recognizer: {self.config.AZURE_FORM_RECOGNIZER_ENDPOINT}")
            print(f"   Azure Search: {self.config.AZURE_SEARCH_ENDPOINT}")
            print(f"   Search Index: {self.config.AZURE_SEARCH_INDEX_NAME}")
            print(f"   OpenAI Model: {self.config.OPENAI_MODEL}")
            if self.config.OPENAI_BASE_URL:
                print(f"   OpenAI Base URL: {self.config.OPENAI_BASE_URL}")
            else:
                print(f"   OpenAI Base URL: Standard OpenAI API")
            
            # Create agent with debugging enabled
            from lease_agent import create_lease_agent_from_config
            self.agent = create_lease_agent_from_config(self.config)
            
            # Update agent settings from config
            self.agent.agent_executor.verbose = self.config.AGENT_VERBOSE
            self.agent.agent_executor.max_iterations = self.config.AGENT_MAX_ITERATIONS
            
        except Exception as e:
            print(f"\n❌ Failed to initialize agent: {e}")
            print(f"\n🔍 Debug Information:")
            print(f"   Error Type: {type(e).__name__}")
            print(f"   Error Details: {str(e)}")
            
            # Check for common connection issues
            if "connection" in str(e).lower() or "timeout" in str(e).lower():
                print(f"\n💡 Possible connection issues:")
                print(f"   - Check your internet connection")
                print(f"   - Verify Azure/OpenAI service endpoints are correct")
                print(f"   - Ensure API keys are valid and have proper permissions")
                print(f"   - Check if services are running and accessible")
            
            if "authentication" in str(e).lower() or "unauthorized" in str(e).lower():
                print(f"\n🔑 Authentication issues:")
                print(f"   - Verify API keys are correct and not expired")
                print(f"   - Check if keys have necessary permissions")
                print(f"   - Ensure endpoints match the region of your keys")
            
            sys.exit(1)
    
    def _initialize_chat_history(self):
        """Initialize chat history"""
        self.chat_history = ChatHistory(
            self.config.CHAT_HISTORY_FILE,
            self.config.MAX_CHAT_HISTORY
        )
    
    def _print_welcome(self):
        """Print welcome message"""
        print("\n" + "="*60)
        print("🤖 LEASE DOCUMENT PROCESSING AGENT - CHAT INTERFACE")
        print("="*60)
        print("Welcome! I can help you with lease document processing.")
        print("\n🔧 Available capabilities:")
        print("   📄 Process PDF documents with OCR")
        print("   🏢 Find documents by location")
        print("   🔍 Search and extract specific fields")
        print("   📊 Analyze and summarize document collections")
        print("   💬 Answer questions about your lease documents")
        
        print(f"\n💡 Quick commands:")
        print("   /help     - Show detailed help")
        print("   /history  - Show recent chat history")
        print("   /status   - Show system status")
        print("   /clear    - Clear chat history")
        print("   /quit     - Exit the chat")
        
        print(f"\n⚙️  Settings:")
        print(f"   Model: {self.config.OPENAI_MODEL}")
        print(f"   Verbose: {self.config.AGENT_VERBOSE}")
        print(f"   Max iterations: {self.config.AGENT_MAX_ITERATIONS}")
        
        print("\n" + "-"*60)
        print("💬 Start chatting! Type your questions or commands below.")
        print("-"*60 + "\n")
    
    def _print_help(self):
        """Print detailed help information"""
        help_text = """
🆘 LEASE DOCUMENT AGENT - HELP

📋 WHAT CAN I DO?
• Process PDF lease documents with OCR text extraction
• Store documents in vector database with embeddings
• Search for documents by location, dates, or content
• Extract specific fields from lease agreements
• Answer questions about your document collection
• Generate summaries and reports

💬 EXAMPLE QUESTIONS & COMMANDS:
• "Process the lease document at /path/to/lease.pdf"
• "Find all documents for locations in Chicago"
• "What office leases expire in 2024?"
• "Extract building areas from all warehouse leases"
• "Show me lease documents for ABC Corporation"
• "Generate a summary of all lease documents"

🎯 QUICK COMMANDS:
• /help     - Show this help message
• /history  - Display recent chat history
• /status   - Show system and configuration status
• /clear    - Clear chat history
• /examples - Show more example queries
• /config   - Show current configuration
• /quit     - Exit the chat interface

🔧 HOW IT WORKS:
The agent uses a ReAct (Reasoning + Acting) approach:
1. Analyzes your question
2. Plans the necessary steps
3. Uses specialized tools (OCR, vector search, etc.)
4. Combines results to answer your question

🛠️  AVAILABLE TOOLS:
• Azure OCR Extractor - Extract text from PDF files
• Vector Store Ingest - Add documents to searchable database
• Location Matcher - Find documents by location
• Vector Search - Semantic search across documents

💡 TIPS:
• Be specific in your questions for better results
• You can ask follow-up questions about previous results
• The agent remembers context within the conversation
• Use natural language - no need for technical syntax
        """
        print(help_text)
    
    def _print_examples(self):
        """Print example queries"""
        examples_text = """
📝 EXAMPLE QUERIES

🏢 LOCATION-BASED QUERIES:
• "Find all lease documents for properties in downtown Seattle"
• "Show me documents with addresses containing 'Main Street'"
• "What leases do we have in the financial district?"

📅 DATE-BASED QUERIES:
• "Which leases expire between January 2024 and December 2024?"
• "Find all lease agreements that started in 2023"
• "Show me leases with terms longer than 5 years"

📊 DATA EXTRACTION:
• "Extract location, start date, and building area from all office leases"
• "What's the average building area across all lease documents?"
• "List all different building types in our lease portfolio"

🔍 SEARCH & ANALYSIS:
• "Find lease documents mentioning 'parking spaces'"
• "Which client has the most lease agreements?"
• "Show me all retail space leases under 5000 sq ft"

📋 DOCUMENT PROCESSING:
• "Process the PDF file at ../leases/client1/lease_agreement.pdf"
• "Add all documents in the ../leases/new_client/ folder to the database"
• "Extract text from lease.pdf and store it with metadata"

📈 REPORTING:
• "Generate a summary report of all lease documents"
• "Compare lease terms across different property types"
• "Find leases that need renewal in the next 6 months"
        """
        print(examples_text)
    
    def _print_status(self):
        """Print system status"""
        print(f"\n📊 SYSTEM STATUS")
        print("-" * 30)
        print(f"🕒 Session started: {self.session_start.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"💬 Messages in history: {len(self.chat_history.messages)}")
        print(f"🔧 Agent verbose mode: {self.config.AGENT_VERBOSE}")
        print(f"🎯 Max agent iterations: {self.config.AGENT_MAX_ITERATIONS}")
        print(f"🗂️  Default input folder: {self.config.DEFAULT_INPUT_FOLDER}")
        print(f"📁 Search index: {self.config.AZURE_SEARCH_INDEX_NAME}")
        
        # Test agent connectivity
        try:
            test_result = self.agent.ask_question("Hello, are you working?")
            if test_result.get('status') == 'success':
                print("✅ Agent connectivity: OK")
            else:
                print("⚠️  Agent connectivity: Issues detected")
        except Exception as e:
            print(f"❌ Agent connectivity: Error - {e}")
    
    def _show_history(self, count: int = 10):
        """Show recent chat history"""
        recent_messages = self.chat_history.get_recent_messages(count)
        
        if not recent_messages:
            print("📝 No chat history available")
            return
        
        print(f"\n📝 RECENT CHAT HISTORY (last {min(count, len(recent_messages))} messages)")
        print("-" * 50)
        
        for msg in recent_messages:
            timestamp = datetime.fromisoformat(msg['timestamp']).strftime('%H:%M:%S')
            role_symbol = "👤" if msg['role'] == 'user' else "🤖"
            print(f"{role_symbol} [{timestamp}] {msg['content'][:100]}")
            if len(msg['content']) > 100:
                print("   ...")
        print()
    
    def _handle_command(self, command: str) -> bool:
        """Handle special commands. Returns True if should continue, False to exit"""
        
        command = command.lower().strip()
        
        if command == '/quit' or command == '/exit':
            return False
        
        elif command == '/help':
            self._print_help()
        
        elif command == '/examples':
            self._print_examples()
        
        elif command == '/status':
            self._print_status()
        
        elif command == '/history':
            self._show_history()
        
        elif command.startswith('/history'):
            # Handle /history N to show N messages
            parts = command.split()
            count = 10
            if len(parts) > 1:
                try:
                    count = int(parts[1])
                except ValueError:
                    print("⚠️  Invalid number. Usage: /history [number]")
                    return True
            self._show_history(count)
        
        elif command == '/clear':
            self.chat_history.clear_history()
            print("🧹 Chat history cleared")
        
        elif command == '/config':
            self.config.print_status()
        
        else:
            print(f"❓ Unknown command: {command}")
            print("💡 Type /help for available commands")
        
        return True
    
    def _process_user_input(self, user_input: str) -> Optional[Dict[str, Any]]:
        """Process user input and get agent response"""
        
        print("🤖 Thinking...")
        start_time = time.time()
        
        try:
            # Get response from agent
            result = self.agent.ask_question(user_input)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Add metadata
            result['processing_time'] = processing_time
            result['timestamp'] = datetime.now().isoformat()
            
            return result
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'processing_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            }
    
    def run_chat(self):
        """Main chat loop"""
        
        self._print_welcome()
        
        try:
            while True:
                # Get user input
                try:
                    user_input = input("👤 You: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\n\n👋 Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    should_continue = self._handle_command(user_input)
                    if not should_continue:
                        break
                    continue
                
                # Save user message
                self.chat_history.add_message('user', user_input)
                
                # Process with agent
                result = self._process_user_input(user_input)
                
                if result:
                    # Display response
                    if result.get('status') == 'success':
                        response = result.get('result', 'No response')
                        print(f"\n🤖 Agent: {response}")
                        
                        # Show processing time if verbose
                        if self.config.SHOW_TOOL_EXECUTION:
                            processing_time = result.get('processing_time', 0)
                            print(f"⏱️  (Processed in {processing_time:.2f}s)")
                        
                        # Save agent response
                        self.chat_history.add_message('assistant', response, {
                            'processing_time': result.get('processing_time'),
                            'status': 'success'
                        })
                    
                    else:
                        error_msg = result.get('error', 'Unknown error')
                        print(f"\n❌ Error: {error_msg}")
                        
                        # Save error
                        self.chat_history.add_message('assistant', f"Error: {error_msg}", {
                            'processing_time': result.get('processing_time'),
                            'status': 'error'
                        })
                
                print()  # Empty line for readability
                
        except KeyboardInterrupt:
            print("\n\n👋 Chat interrupted. Goodbye!")
        
        finally:
            # Save chat history
            self.chat_history.save_history()
            print(f"💾 Chat history saved to {self.config.CHAT_HISTORY_FILE}")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Interactive chat with the Lease Document Agent')
    parser.add_argument('--config', help='Path to configuration file')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose mode')
    parser.add_argument('--no-history', action='store_true', help='Don\'t load/save chat history')
    
    args = parser.parse_args()
    
    try:
        # Create chat interface
        chat = LeaseAgentChat(config_file=args.config)
        
        # Override settings from command line
        if args.verbose:
            chat.config.AGENT_VERBOSE = True
            chat.config.SHOW_TOOL_EXECUTION = True
        
        if args.no_history:
            chat.config.CHAT_HISTORY_FILE = "/dev/null"  # Disable history
        
        # Start chat
        chat.run_chat()
        
    except Exception as e:
        print(f"❌ Failed to start chat: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()