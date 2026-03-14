FROM python:3.12-slim

WORKDIR /app

# gcc нужен если какой-то wheel собирается из исходников
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

COPY requirements.ml.txt .
RUN pip install --no-cache-dir -r requirements.ml.txt

COPY ml_service.py .

RUN mkdir -p models

EXPOSE 8000

ENV PYTHONUNBUFFERED=1
CMD ["python", "-u", "ml_service.py"]
