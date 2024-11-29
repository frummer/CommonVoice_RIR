# Use Python 3.12.6 as the base image
FROM python:3.12.6-slim

# Set the working directory in the container
WORKDIR /app

# Upgrade pip to the latest version
RUN pip install --upgrade pip --root-user-action=ignore

# Copy requirements.txt to the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --root-user-action=ignore -r requirements.txt

# Copy the source code into the container
COPY src/ ./src

# Set the default command to run Python
CMD ["python"]