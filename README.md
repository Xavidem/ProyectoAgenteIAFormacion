# AGENTE DE INTELIGENCIA ARTIFICIAL FORMACIÓN QA
Este agente es un chatbot que permite al usuario escribir por texto su petición de encontrar información sobre un tema concreto (p.e. Gherkin o Cucumber) y es capaz de mostrarle documentos/vídeos relacionados a ese tema. 
En caso de que no encuentre la información necesaria en los ficheros del sistema, hará un fallback a web para encontrar webs relacionadas a ese tema.

## FLUJO DE INTERACCIÓN DEL AGENTE
El agente tendrá el siguiente flujo de interacción con el usuario, con la base de datos y con la web:


## FASES DEL PROYECTO

### Instalar dependencias para el fichero metadata_extractor.py
pip install pymupdf nltk tqdm python-docx python-pptx 

### Instalar dependencias para el fichero text_extractor.py
pip install qdrant_client tqdm sentence_transformers

### Instalar dependencias para el fichero llm_service.py
pip install pydantic fastapi

### Levantar el Podman por primera vez
podman run -d --name qdrant -p 6333:6333 -v qdrant_data:/qdrant/storage qdrant/qdrant:latest


# Como reconstruir el podman-compose.yml si se hace algun cambio
podman-compose down
podman-compose build llava-service
podman-compose build chat-api
podman-compose up -d


# Algunas llamadas a la API
curl.exe -X POST http://localhost:8001/infer -H "Content-Type: application/json" -d @payload.json

# Reconstruir imagen de llava-service o cualquiera
podman build -t llava-service -f Containerfile . 
podman rm -f llava-service
podman run -d --name llava-service -p 8001:8001 llava-service:latest