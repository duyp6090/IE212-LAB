#!/bin/bash

echo "Setting up project in WSL..."

# Create project directory in WSL
mkdir -p ~/lab4

# Copy project 
cp -r /mnt/d/Study/Nam_Bon/Big_Data/Lab/lab4_code/IE212.Q11/* ~/lab4/

# Change to project directory
cd ~/lab4

echo "Installing Library"

sudo apt update
sudo apt install -y python3 python3-pip python3-venv python3-dev build-essential

python3 -m venv venv

source venv/bin/activate

pip install --upgrade pip setuptools wheel
pip install "numpy<2.0"
pip install opencv-python>=4.8.0
pip install mediapipe>=0.10.13
pip install pyspark==3.4.3
pip install py4j==0.10.9.7
pip install pytest>=7.0.0