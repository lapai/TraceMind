# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
# PYTHONUNBUFFERED=1 确保日志实时输出，方便 NAS 查看日志调试
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

# Set the working directory in the container
WORKDIR /app

# Install system dependencies if needed
# (Currently not needed for the simple requirements, but good for future-proofing)
# RUN apt-get update && apt-get install -y --no-install-recommends gcc python3-dev && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# Copy the rest of the application code into the container
# .dockerignore should exclude data files and __pycache__
COPY . .

# Ensure static directory exists
RUN mkdir -p static data

# Make port 8000 available
EXPOSE 8000

# Run the application
# We use python main.py instead of uvicorn to trigger the port detection and setup logic
CMD ["python", "main.py"]
