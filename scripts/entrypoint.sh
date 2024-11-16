#!/bin/bash

set -e

case "$1" in
    "train")
        echo "Starting training..."
        python3 main.py --mode train
        ;;
    "api")
        echo "Starting API server..."
        uvicorn api.inference:app --host 0.0.0.0 --port 8000 --workers 4
        ;;
    "monitor")
        echo "Starting monitoring service..."
        python3 main.py --mode monitor
        ;;
    *)
        echo "Unknown command: $1"
        echo "Usage: $0 {train|api|monitor}"
        exit 1
        ;;
esac