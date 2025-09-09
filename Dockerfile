FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies (if any compiled deps are needed in requirements)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (leverage Docker layer caching)
COPY req.txt ./
RUN pip install --no-cache-dir -r req.txt

# Copy project files
COPY . .

# Expose the Flask port
EXPOSE 5000

# Start the application
CMD ["python", "run.py"]


