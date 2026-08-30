#!/bin/bash

# M2T2 Demo Launcher - Sets up tmux session with meshcat-server, m2t2_server, and client demo

set -e  # Exit on error

# Configuration
SESSION_NAME="m2t2-demo"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Error handling function
error_exit() {
    echo -e "${RED}Error: $1${NC}" >&2
    exit 1
}

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    error_exit "tmux is not installed. Please install it first (apt install tmux / brew install tmux)"
fi

# Check if pixi is available
if ! command -v pixi &> /dev/null; then
    error_exit "pixi is not installed. Please install it first (curl -fsSL https://pixi.sh/install.sh | bash)"
fi

# Check if pixi.toml exists
if [[ ! -f "${SCRIPT_DIR}/pixi.toml" ]]; then
    error_exit "pixi.toml not found. Please run from the M2T2 directory."
fi

# Check if required Python files exist
if [[ ! -f "${SCRIPT_DIR}/m2t2_server.py" ]]; then
    error_exit "m2t2_server.py not found in ${SCRIPT_DIR}"
fi

if [[ ! -f "${SCRIPT_DIR}/m2t2_client_demo.py" ]]; then
    error_exit "m2t2_client_demo.py not found in ${SCRIPT_DIR}"
fi

# Check if session already exists
if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo -e "${YELLOW}Warning: Session '${SESSION_NAME}' already exists.${NC}"
    read -p "Do you want to kill it and create a new one? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        tmux kill-session -t "${SESSION_NAME}"
        echo -e "${GREEN}Killed existing session.${NC}"
    else
        echo -e "${GREEN}Attaching to existing session...${NC}"
        tmux attach-session -t "${SESSION_NAME}"
        exit 0
    fi
fi

echo -e "${GREEN}Creating tmux session '${SESSION_NAME}'...${NC}"

# Create new tmux session (detached) and run client demo in the first pane
tmux new-session -d -s "${SESSION_NAME}" -c "${SCRIPT_DIR}" || error_exit "Failed to create tmux session"

# Set up the first pane (m2t2_client_demo.py with pixi)
tmux send-keys -t "${SESSION_NAME}:0.0" "echo -e '${GREEN}Starting M2T2 client demo...${NC}'" C-m
tmux send-keys -t "${SESSION_NAME}:0.0" "sleep 3 && echo 'Waiting for server to initialize...' && sleep 2" C-m
tmux send-keys -t "${SESSION_NAME}:0.0" "pixi run python m2t2_client_demo.py sample_data/real_world/00" C-m

# Split window horizontally to create second pane
tmux split-window -h -t "${SESSION_NAME}:0" -c "${SCRIPT_DIR}"

# Set up the second pane (m2t2_server.py with pixi)
tmux send-keys -t "${SESSION_NAME}:0.1" "echo -e '${GREEN}Starting M2T2 server...${NC}'" C-m
tmux send-keys -t "${SESSION_NAME}:0.1" "pixi run python m2t2_server.py" C-m

# Split the second pane vertically to create third pane
tmux split-window -v -t "${SESSION_NAME}:0.1" -c "${SCRIPT_DIR}"

# Set up the third pane (meshcat-server)
tmux send-keys -t "${SESSION_NAME}:0.2" "echo -e '${GREEN}Starting meshcat-server...${NC}'" C-m
tmux send-keys -t "${SESSION_NAME}:0.2" "pixi run meshcat-server" C-m

# Adjust pane layout for better visibility
tmux select-layout -t "${SESSION_NAME}:0" main-vertical

# Select the first pane
tmux select-pane -t "${SESSION_NAME}:0.0"

echo -e "${GREEN}Session created successfully!${NC}"
echo -e "Layout:"
echo -e "  - Pane 0: m2t2_client_demo.py"
echo -e "  - Pane 1: m2t2_server.py"
echo -e "  - Pane 2: meshcat-server"
echo ""
echo -e "${GREEN}Attaching to session...${NC}"
echo -e "${YELLOW}Tip: Use Ctrl+B then D to detach from the session${NC}"

# Attach to the session
tmux attach-session -t "${SESSION_NAME}"

# Kill the session after detaching
echo -e "${GREEN}Killing session '${SESSION_NAME}'...${NC}"
tmux kill-session -t "${SESSION_NAME}"
