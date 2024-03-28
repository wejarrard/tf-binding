#!/bin/bash
# entrypoint.sh
set -e

# Check if the first argument is `serve`
if [ "$1" = "serve" ]; then
    # Start the FastAPI app with Uvicorn
    uvicorn main:app --host 0.0.0.0 --port 8080
else
    # Default or other custom commands can be handled here
    exec "$@"
fi
