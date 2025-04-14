#!/bin/bash

# Function to kill a process based on a search string, excluding tmux server itself!
kill_process_by_string() {
    local SEARCH_STRING="$1"

    # Find the process IDs based on the search string, excluding tmux server itself!
    # local PROCESS_PIDS=$(pgrep -f "$SEARCH_STRING")
    local PROCESS_PIDS=$(pgrep -a -f "$SEARCH_STRING" | grep -v "tmux" | awk '{print $1}')

    # Check if any process IDs were found
    if [[ -z "$PROCESS_PIDS" ]]; then
        echo -e "\e[33mProcess with search string '$SEARCH_STRING' not found.\e[0m"
    else
        # Kill each process
        for PROCESS_PID in $PROCESS_PIDS; do
            kill -9 $PROCESS_PID
            echo -e "\e[32mProcess (PID: $PROCESS_PID) with search string '$SEARCH_STRING' has been killed.\e[0m"
        done
    fi
}

# Kill Streamlit server
kill_process_by_string "poetry run streamlit run streamlit_app.py"
