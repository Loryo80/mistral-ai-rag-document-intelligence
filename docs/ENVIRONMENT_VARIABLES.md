# Enhanced Legal AI System - Environment Variables

## 📋 Overview

This document provides the definitive guide for environment variables required by the Enhanced Legal AI System. The requirements have been verified against the working application and updated to reflect actual usage.

## ✅ Required Environment Variables

### Database Configuration
```bash
# Required for all database operations
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_supabase_key

# Note: SUPABASE_ANON_KEY is NOT required
# The application works with just SUPABASE_KEY which provides sufficient access
```

### AI Services Configuration
```bash
# Required for core functionality (OCR, embeddings, chat)
MISTRAL_API_KEY=your_mistral_api_key

# Optional for premium features (advanced models)
OPENAI_API_KEY=your_openai_api_key
```

## 🔧 Optional Environment Variables

### Processing Configuration
```bash
# AI Provider Settings (defaults work fine)
LLM_PROVIDER=openai                    # Options: openai, mistral
LLM_MODEL=gpt-4o-mini                  # OpenAI model to use
MISTRAL_OCR_MODEL=pixtral-12b-2409     # Mistral OCR model
EMBEDDING_MODEL=mistral-embed          # Embedding model

# Application Behavior
SKIP_DB_USER_CREATION=false            # Skip user creation in database
DEBUG_MODE=false                       # Enable debug logging
LOG_LEVEL=INFO                         # Logging level
```

### Server Configuration
```bash
# Port Configuration (defaults to standard ports)
STREAMLIT_SERVER_PORT=8501             # Primary interface port
ENHANCED_SERVER_PORT=8506              # Enhanced interface port
ADMIN_SERVER_PORT=8507                 # Admin dashboard port
```

## 📝 Environment File Template

Create a `.env` file in your project root:

```bash
# Essential Configuration (Minimum required)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_supabase_service_role_key
MISTRAL_API_KEY=your_mistral_api_key

# Optional Premium Features
OPENAI_API_KEY=your_openai_api_key

# Optional Customization
LLM_MODEL=gpt-4o-mini
DEBUG_MODE=false
```

## ✅ Verification Script

Test your environment configuration:

```python
#!/usr/bin/env python3
"""
Environment Variables Verification Script
"""
import os
import sys

def verify_environment():
    """Verify all required environment variables are set."""
    
    # Required variables
    required_vars = {
        'SUPABASE_URL': 'Database connection URL',
        'SUPABASE_KEY': 'Database access key',
        'MISTRAL_API_KEY': 'Mistral AI API key'
    }
    
    # Optional variables
    optional_vars = {
        'OPENAI_API_KEY': 'OpenAI API key (for premium features)',
        'LLM_MODEL': 'Language model selection',
        'DEBUG_MODE': 'Debug mode toggle'
    }
    
    print("🔍 Enhanced Legal AI System - Environment Verification")
    print("=" * 60)
    
    # Check required variables
    missing_required = []
    print("\n✅ Required Variables:")
    for var, description in required_vars.items():
        value = os.getenv(var)
        if value:
            print(f"  ✅ {var}: {'*' * min(len(value), 20)} ({description})")
        else:
            print(f"  ❌ {var}: MISSING ({description})")
            missing_required.append(var)
    
    # Check optional variables
    print("\n🔧 Optional Variables:")
    for var, description in optional_vars.items():
        value = os.getenv(var)
        if value:
            print(f"  ✅ {var}: {value} ({description})")
        else:
            print(f"  ⚪ {var}: Not set ({description})")
    
    # Summary
    print("\n" + "=" * 60)
    if missing_required:
        print(f"❌ Missing {len(missing_required)} required variables: {', '.join(missing_required)}")
        print("   Please set these variables before running the application.")
        return False
    else:
        print("✅ All required environment variables are set!")
        print("🚀 The Enhanced Legal AI System is ready to run.")
        return True

if __name__ == "__main__":
    success = verify_environment()
    sys.exit(0 if success else 1)
```

## 🚀 Quick Setup Commands

### For Development
```bash
# Create .env file with essential variables
cat > .env << EOF
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_supabase_key
MISTRAL_API_KEY=your_mistral_api_key
OPENAI_API_KEY=your_openai_api_key
LLM_MODEL=gpt-4o-mini
DEBUG_MODE=false
EOF

# Verify configuration
python verify_environment.py

# Run the application
python -m streamlit run src/app/app.py
```

### For Production
```bash
# Set production environment variables
export SUPABASE_URL="https://your-project.supabase.co"
export SUPABASE_KEY="your_production_supabase_key"
export MISTRAL_API_KEY="your_production_mistral_key"
export OPENAI_API_KEY="your_production_openai_key"
export DEBUG_MODE="false"

# Verify and run
python verify_environment.py && streamlit run src/app/app.py --server.port 8501
```

## 🔒 Security Best Practices

### Key Management
- **Never commit API keys** to version control
- **Use environment-specific keys** (dev/staging/prod)
- **Rotate keys regularly** for production environments
- **Use secrets management** in cloud deployments

### Access Control
- **SUPABASE_KEY**: Use service role key for full access
- **Database Security**: Ensure RLS policies are enabled
- **API Rate Limits**: Monitor usage to prevent abuse

## 🚨 Troubleshooting

### Common Issues

#### Database Connection Failed
```bash
❌ Error: Invalid API key or URL
```
**Solution**: Verify SUPABASE_URL and SUPABASE_KEY are correct

#### Mistral API Issues
```bash
❌ Error: Mistral API authentication failed
```
**Solution**: Check MISTRAL_API_KEY is valid and has credits

#### Application Won't Start
```bash
❌ Missing required environment variables
```
**Solution**: Run the verification script to identify missing variables

### Validation Commands
```bash
# Test database connection
python -c "from src.storage.db import Database; db = Database(); print('✅ Database connected')"

# Test Mistral API
python -c "import requests; import os; r = requests.get('https://api.mistral.ai/v1/models', headers={'Authorization': f'Bearer {os.getenv(\"MISTRAL_API_KEY\")}'}); print(f'Mistral API: {r.status_code}')"

# Test environment loading
python -c "import os; print('✅ Environment loaded') if os.getenv('SUPABASE_URL') else print('❌ Environment not loaded')"
```

## 📊 Configuration Status

### Current System Status (June 2025)
- ✅ **Database**: Working with SUPABASE_KEY only (no ANON_KEY needed)
- ✅ **Mistral API**: Fully operational for OCR, embeddings, and chat
- ✅ **Chunks/Embeddings**: Tables operational with existing data
- ✅ **Vector Search**: Working with real document data
- ⚪ **OpenAI**: Optional for premium features

### Verified Configurations
- **Minimal Setup**: SUPABASE_URL + SUPABASE_KEY + MISTRAL_API_KEY
- **Full Setup**: Above + OPENAI_API_KEY for premium features
- **Production**: Above + proper logging and debug settings

---

**Environment Guide Version**: v2.3.0 - Verified Configuration  
**Last Updated**: June 2025  
**Status**: Production Ready with Verified Requirements  
**Testing**: All configurations validated against working app.py  

🎉 **This environment configuration guide reflects the actual working requirements of the Enhanced Legal AI System!** 