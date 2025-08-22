# ======================
# Worker Dockerfile
# ======================
FROM python:3.11-slim

WORKDIR /worker

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run refresh script in loop (simulating cron every 6h)
CMD ["sh", "-c", "while true; do python refresh.py; sleep 21600; done"]
