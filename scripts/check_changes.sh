#!/bin/bash

# Function to check each file
check_file() {
    local file=$1
    # Extract changes that were added in the given file
    local changes=$(git diff $file | grep '^+' | grep -v '+++')

    # Check if all changes are on lines containing allowed changes
    if echo "$changes" | grep -v -E '"latencyMs":|"totalLatencyMs":|"reason":|"value":|"score":|"pass":|"assertPassCount":|"assertFailCount":|"success":|"successes":|"failures":|"testPassCount":|"testFailCount":|"text":|exec:python'; then
        # Print the file name with an error message
        echo "Error: File '$file' has potentially wrong changes on lines"
        return 1
    fi
    return 0
}

# Initialize status
status=0

# Check for unstaged changes
if ! git diff --quiet '**/latest.json'; then
    # Loop through each file that has changes
    for file in $(git diff --name-only '**/latest.json'); do
        check_file "$file" || status=1
    done

    # Exit with status based on checks
    if [ $status -ne 0 ]; then
        exit 1
    else
        echo "All changes are on lines containing allowed words."
        exit 0
    fi
else
    echo "No changes detected."
    exit 0
fi

