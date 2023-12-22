FROM ubuntu:22.04

WORKDIR /app

# Install Python and pip
RUN apt-get update && apt-get install -y python3-pip python3 && \
    rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN groupadd -g 10001 user && \
    useradd -u 10000 -g user user && \
    chown -R user:user /app

# Copy requirements.txt first to leverage Docker cache
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Environment variable
ENV PYTHONPATH=/app

# Switch to non-root user
USER user
