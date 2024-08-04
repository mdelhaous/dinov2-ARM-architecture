# Use a compatible base image for ARM64
FROM --platform=linux/arm64/v8 ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

# Install required system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-venv \
    python3.9-dev \
    python3-pip \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.9 as the default Python version
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1

# Upgrade pip
RUN python3.9 -m pip install --upgrade pip

# Copy the requirements file into the container
COPY requirements-arm64.txt requirements.txt

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Set the environment variable to disable xformers
ENV XFORMERS_DISABLED=1

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "API:app", "--host", "0.0.0.0", "--port", "8000"]
