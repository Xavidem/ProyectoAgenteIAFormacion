# Compilar llama.cpp
FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git libgomp1 libcurl4-openssl-dev && rm -r /var/lib/apt/lists/*

# Dependecias de Python
RUN pip install --no-cache-dir fastapi uvicorn llama-cpp-python

WORKDIR /app

# Copiamos nuestro servicio y el modelo
COPY src/service/llm_service.py /app/llm_service.py
COPY models/llava/ggml-model-q4_k.gguf /app/models/ggml-model-q4_k.gguf

# Hacemos EXPOSE del puerto
EXPOSE 8001

# Comandos por defecto
CMD ["uvicorn", "llm_service:app", "--host", "0.0.0.0", "--port", "8001"]