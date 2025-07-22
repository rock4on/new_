#!/usr/bin/env python3
"""
Setup script for the Lease Document Processing Agent
Helps users get started with configuration and dependencies
"""

import sys
import subprocess
from pathlib import Path
from config import Config


def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing required dependencies...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False


def create_sample_config():
    """Create sample configuration file"""
    print("📝 Creating sample configuration file...")
    
    try:
        Config.create_sample_config_file()
        print("✅ Sample configuration created!")
        print("💡 Next step: Copy config_sample.py to config_local.py and fill in your values")
        return True
    except Exception as e:
        print(f"❌ Failed to create sample config: {e}")
        return False


def check_configuration():
    """Check if configuration is complete"""
    print("🔧 Checking configuration...")
    
    try:
        from config import get_config
        
        config = get_config()
        is_valid, missing_fields = config.validate()
        
        if is_valid:
            print("✅ Configuration is complete!")
            return True
        else:
            print("⚠️  Configuration is incomplete!")
            print("Missing fields:")
            for field in missing_fields:
                print(f"   - {field}")
            return False
            
    except Exception as e:
        print(f"❌ Error checking configuration: {e}")
        return False


def test_agent():
    """Test if the agent can be initialized"""
    print("🧪 Testing agent initialization...")
    
    try:
        from lease_agent import create_lease_agent_from_env
        from config import get_config
        
        config = get_config()
        is_valid, _ = config.validate()
        
        if not is_valid:
            print("⚠️  Cannot test agent - configuration incomplete")
            return False
        
        # Try to create agent (this will test all connections)
        from lease_agent import create_lease_agent_from_config
        agent = create_lease_agent_from_config(config)
        
        print("✅ Agent initialized successfully!")
        print("🎉 You're ready to start chatting!")
        return True
        
    except Exception as e:
        print(f"❌ Agent initialization failed: {e}")
        print("💡 Check your configuration values and network connectivity")
        return False


def main():
    """Main setup function"""
    print("🤖 LEASE DOCUMENT AGENT SETUP")
    print("=" * 40)
    
    # Step 1: Install dependencies
    if not install_dependencies():
        print("❌ Setup failed at dependency installation")
        return False
    
    print()
    
    # Step 2: Create sample configuration
    if not Path("config_local.py").exists():
        if not create_sample_config():
            print("❌ Setup failed at configuration creation")
            return False
        
        print()
        print("🛑 SETUP PAUSED")
        print("=" * 20)
        print("Next steps:")
        print("1. Copy config_sample.py to config_local.py")
        print("2. Edit config_local.py and fill in your actual values:")
        print("   - Azure Form Recognizer endpoint and key")
        print("   - OpenAI API key")
        print("   - Azure AI Search endpoint and key")
        print("3. Run this setup script again: python setup.py")
        print()
        print("💡 You can find these values in your Azure portal and OpenAI dashboard")
        return True
    
    # Step 3: Check configuration
    if not check_configuration():
        print("❌ Please complete your configuration in config_local.py")
        return False
    
    print()
    
    # Step 4: Test agent
    if not test_agent():
        print("❌ Setup completed but agent test failed")
        print("💡 Check your configuration values and try again")
        return False
    
    print()
    print("🎉 SETUP COMPLETED SUCCESSFULLY!")
    print("=" * 30)
    print("You can now:")
    print("• Run the chat interface: python chat_agent.py")
    print("• Use the example script: python example_usage.py")
    print("• Import the agent in your own code")
    print()
    print("💬 Quick start: python chat_agent.py")
    
    return True


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)