FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git libgomp1 libcurl4-openssl-dev curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements/requirements.llava.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY src/service/llm_service.py /app/llm_service.py

EXPOSE 8001

CMD ["uvicorn", "llm_service:app", "--host", "0.0.0.0", "--port", "8001"]
