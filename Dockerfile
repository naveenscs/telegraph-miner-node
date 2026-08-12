FROM python:3.10-slim

WORKDIR /app

# Prevent Python from writing .pyc files & enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy repository files into the container
COPY . .

EXPOSE 8000

# Launch Uvicorn server pointing to src/miner.py
CMD ["python3", "-m", "uvicorn", "src.miner:app", "--host", "0.0.0.0", "--port", "8000"]
