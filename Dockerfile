FROM nikolaik/python-nodejs:python3.12-nodejs20-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y make && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Build frontend
WORKDIR /app/frontend
# Add a dummy build command or just standard npm run build
RUN npm install && npm run build

# Generate dataset and train models
WORKDIR /app
RUN python scripts/perfect_dataset.py && python src/train.py

# Expose port
EXPOSE 8000

# Start FastAPI server
CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
