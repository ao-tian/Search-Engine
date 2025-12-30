#!/bin/bash
# Setup script for Search Engine project
# Creates virtual environment and installs dependencies

echo "Setting up Search Engine project..."
echo ""

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "Setup complete!"
echo ""
echo "To activate the virtual environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "Then you can run:"
echo "  python3 run.py          # Start web interface"
echo "  python3 vector_search.py  # Run CLI version"
echo ""
echo "To deactivate, run:"
echo "  deactivate"

