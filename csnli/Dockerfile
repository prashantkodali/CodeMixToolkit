# Use Python 3.7 as base image
FROM python:3.7-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libenchant-2-2 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files
COPY requirements_from_conda.txt .

# Run kenlm
RUN pip install https://github.com/kpu/kenlm/archive/master.zip

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_from_conda.txt

# Copy all files
COPY . .


RUN python build_viterbi.py build_ext --inplace

# Expose the port the app runs on
EXPOSE 6000

# Command to run the application
CMD ["python", "csnli_api.py"] 