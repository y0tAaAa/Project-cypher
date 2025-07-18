#!/bin/bash

# Initialize conda for shell
source ~/miniconda3/etc/profile.d/conda.sh

# Set project directory
export PROJECT_DIR=~/Project-cypher

# First, move to home directory
cd ~

# Remove existing project directory if it exists
rm -rf $PROJECT_DIR

# Clone the repository
git clone https://github.com/y0tAaAa/Project-cypher.git $PROJECT_DIR

# Enter project directory
cd $PROJECT_DIR || exit 1

# Remove existing conda environment if it exists
conda deactivate
conda env remove -n cypher -y

# Create and activate conda environment
conda create -n cypher python=3.12 -y
conda activate cypher

# Install dependencies
pip install -r requirements.txt || exit 1

# Create directories for results and models if they don't exist
mkdir -p results
mkdir -p models/multilingual

# Download initial model config if needed
if [ ! -f models/multilingual/config.json ]; then
    python -c "from transformers import AutoConfig; config = AutoConfig.from_pretrained('EleutherAI/gpt-neo-2.7B'); config.save_pretrained('models/multilingual')"
fi

# Kill existing tmux session if it exists
tmux kill-session -t cypher 2>/dev/null || true

# Set up new tmux session
tmux new-session -d -s cypher
tmux send-keys -t cypher "cd $PROJECT_DIR && conda activate cypher" C-m

# Export project directory for other scripts
echo "export PROJECT_DIR=$PROJECT_DIR" >> ~/.bashrc
source ~/.bashrc

echo "Setup complete! Project is installed at: $PROJECT_DIR"
echo "To attach to tmux session, use: tmux attach -t cypher" 