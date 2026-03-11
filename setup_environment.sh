# Create a virtual environment
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create results directory
mkdir -p results

# Print message
echo "Environment setup is complete."