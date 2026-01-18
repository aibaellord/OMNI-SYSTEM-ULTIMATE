
#!/bin/bash
# OMNI-SYSTEM Auto-update script

echo "🔄 Updating development environment..."

# Update Homebrew
brew update && brew upgrade

# Update Python packages
pip install --upgrade pip
pip install --upgrade -r requirements.txt

# Update Node.js packages
npm update -g

# Update Go modules
go mod tidy

# Clean up
brew cleanup
npm cache clean --force

echo "✅ Development environment updated!"
        