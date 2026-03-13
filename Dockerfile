FROM python:3.12-slim

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app source and templates
COPY app.py .
COPY templates/ templates/

# Data directory — mount CSV files here via volume (see docker-compose.yml)
RUN mkdir -p data

EXPOSE 5000

ENV PYTHONUNBUFFERED=1
CMD ["python", "-u", "app.py"]
