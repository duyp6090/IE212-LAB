#!/bin/bash

echo "Starting Spark Processing Server In Linux"

# Change to project directory
cd ~/lab4

# Set JAVA_HOME
export JAVA_HOME=$(dirname $(dirname $(readlink -f $(which java))))
echo "Using Java: $JAVA_HOME"

source venv/bin/activate

echo "Starting Spark Processing Server"
python spark_processing_server.py