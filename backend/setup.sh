#!/bin/bash

# French Exercise App - Backend Quick Setup
# Run this script to install all backend dependencies

echo "Installing Flask and dependencies..."
pip install --upgrade pip setuptools wheel
pip install Flask==3.0.0 Flask-CORS==4.0.0 Flask-SQLAlchemy==3.1.1
pip install python-dotenv requests python-multipart
pip install langchain ollama openai PyPDF2 pydantic
echo "Done! Run: python3 app.py"
