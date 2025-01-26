# Use Python 3.12.6 as the base image
FROM python:3.12.6-slim

# Set the working directory in the container
WORKDIR /app

# Upgrade pip to the latest version
RUN pip install --upgrade pip --root-user-action=ignore

# Install FFmpeg dependencies and FFmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    libopus0 \
    libopus-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


# Copy requirements.txt to the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --root-user-action=ignore -r requirements.txt

# Copy the source code into the container
COPY src/ ./src

# Set Hugging Face token as an environment variable (replace YOUR_HF_API_TOKEN)
ENV HF_TOKEN=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

# Load and preprocess the dataset
RUN python -c "from datasets import load_dataset; \
    import os; \
    dataset = load_dataset('mozilla-foundation/common_voice_12_0', 'ar', split='test', trust_remote_code=True);"

# Set an environment variable for dataset path
ENV DATASET_PATH=/mnt/data

# Set the default command to run Python
CMD ["python"]