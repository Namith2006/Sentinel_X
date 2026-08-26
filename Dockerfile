# Use official Python runtime
FROM python:3.11

# Set the working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Hugging Face Spaces requires running on port 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]