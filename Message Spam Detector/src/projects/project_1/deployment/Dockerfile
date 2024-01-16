# Use Python 3.12 Slim Base Image
FROM python:3.12-slim

# Work Directory
WORKDIR /app

# COPY the Assets
COPY . /app/

# Requirements
RUN pip install -r /app/requirements.txt

# Run Streamlit App
CMD streamlit run /app/app.py --server.port 8501

