# Use official Python runtime
FROM python:3.11

# Create the non-root user required by Hugging Face Spaces
RUN useradd -m -u 1000 user
USER user

# Set environment variables to route PyTorch cache to a writable user folder
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/huggingface

# Set the working directory
WORKDIR $HOME/app

# Copy requirements and install them securely as the user
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy the rest of the project
COPY --chown=user:user . .

# Hugging Face Spaces requires running on port 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]