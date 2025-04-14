#!/bin/bash

SESSION_NAME="llmetrics_webui"

# Store current directory
export CURR_DIR=$(pwd)
# REPO_ROOT_DIR is the parent of the current directory
export REPO_ROOT_DIR=$(dirname "$CURR_DIR")

# Check if the session exists
tmux has-session -t $SESSION_NAME 2>/dev/null

if [ $? != 0 ]; then
    # Session does not exist, create a new one and run the command

    STREAMLIT_BIND_IP="0.0.0.0"
    STREAMLIT_PORT="8501"

    mkdir -p ${REPO_ROOT_DIR}/logs

    cd ${REPO_ROOT_DIR}

    # Create the tmux session and run the command
    cd ${REPO_ROOT_DIR}/webui/streamlit
    tmux new-session -d -s $SESSION_NAME "\
    export STREAMLIT_BIND_IP='${STREAMLIT_BIND_IP}'; \
    export STREAMLIT_PORT='${STREAMLIT_PORT}'; \
    export REPO_ROOT_DIR='${REPO_ROOT_DIR}'; \
    poetry run streamlit run streamlit_app.py --server.address \$STREAMLIT_BIND_IP --server.port \$STREAMLIT_PORT --browser.gatherUsageStats False --server.headless True 2>> \$REPO_ROOT_DIR/webui/logs/llmetrics_webui.log"

    # Check if the session has been created
    tmux has-session -t $SESSION_NAME 2>/dev/null

    if [ $? -eq 0 ]; then
        echo -e "\e[32mCreated new tmux session '$SESSION_NAME'!\e[0m"
    else
        echo -e "\e[31mError: Failed to create tmux session '$SESSION_NAME'!\e[0m"
        exit 1
    fi
else
    echo -e "\e[33mTmux session '$SESSION_NAME' already exists!\e[0m"
fi

cd ${CURR_DIR}

echo "Useful commands:"
echo "> tail -f $REPO_ROOT_DIR/logs/llmetrics_webui.log"
echo "> tmux attach-session -t $SESSION_NAME"
