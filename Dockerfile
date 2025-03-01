# Dockerfile
FROM python:3.9-slim

# Prevent Python from writing .pyc files and enable stdout/stderr logging
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

# Install system dependencies (if needed)
RUN apt-get update && apt-get install -y gcc

# Copy requirements.txt and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy all your application code
COPY . .

# Expose the port your API runs on
EXPOSE 5000

# Use Gunicorn for production server with 2 workers (adjust as needed)
CMD ["gunicorn", "model_api:app", "--bind", "0.0.0.0:5000", "--workers", "2"]
