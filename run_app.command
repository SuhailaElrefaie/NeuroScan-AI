#!/bin/bash

cd "$(dirname "$0")"

echo "Starting NeuroScan AI..."
echo "Current folder:"
pwd

echo ""
echo "Checking Python..."
python3 --version

echo ""
echo "Starting Streamlit..."
python3 -m streamlit run ui.py

echo ""
echo "App closed or an error happened."
echo "Press ENTER to close this window."
read