import json
import fitz
import uuid
from pathlib import Path
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, FieldCondition, MatchValue, Filter
from tqdm import tqdm

"""CONFIGURACION INICIAL"""
# Tamaño del embedding
EMBEDDING_DIM = 384

# Chunking: Tamaño y solapamiento
CHUNK_SIZE = 1000
CHUNK_STRIDE = 600

# Rutas
METADATA_PATH = Path("metadata_master.json")

# Configuracion Qdrant
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "docs"

"""FUNCIONES AUXILIARES"""
#Funcion que limpia el texto
"""DESCRIPCION PRINCIPAL DE LA FUNCION:
OUT: Devuelve el texto sin saltos de linea y normalizando espacios
"""
def clean_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return ''.join(lines)



#Funcion que divide el texto en chunks
"""DESCRIPCION PRINCIPAL DE LA FUNCION:
Divide el texto de un PDF en diferentes chunk con un tamaño maximo por cada uno
OUT: Devuelve una lista de tokens
"""
def chunk_tokens(tokens: list, size: int, stride: int) -> list:
    chunks = []
    for start in range(0, len(tokens), stride):
        chunk = tokens[start:start + size]
        if not chunk:
            break
        chunks.append(chunk)
        if start + size >= len(tokens):
            break
    return chunks

"""INICIALIZACION"""

# Modelo de embeddings
model = SentenceTransformer("./models/all-MiniLM-L6-v2", local_files_only=True)

#Conexion a Qdrant y crea la coleccion
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
if COLLECTION_NAME not in [col.name for col in client.get_collections().collections]:
    client.recreate_collection(collection_name=COLLECTION_NAME,
                               vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE))
    
"""PIPELINE PRINCIPAL"""
def process_and_index():
    records = json.loads(METADATA_PATH.read_text(encoding='utf-8'))

    for rec in tqdm(records, desc="Indexando los documentos"):
        if rec.get("type") != "pdf":
            continue
        doc_id = rec["id"]
        pdf_path = rec["path"]

        # Eliminamos los chunks antiguos de este documento en qdrant
        delete_filter = Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))])
        client.delete(collection_name=COLLECTION_NAME, points_selector=delete_filter)

        # Extraemos el texto
        doc = fitz.open(pdf_path)
        full_text = "".join([page.get_text() for page in doc])
        doc.close()

        # Limpieza del texto
        cleaned = clean_text(full_text)
        tokens = cleaned.split()

        # Generamos los chunks
        chunks = chunk_tokens(tokens, CHUNK_SIZE, CHUNK_STRIDE)

        # Obtenemos los embeddings
        texts = [" ".join(chunk) for chunk in chunks]
        if not texts:
            continue
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

        snippet = " ".join(texts[0].split()[:30])

        # Enviar a Qdrant
        points = []
        for idx, vector in enumerate(embeddings):
            point_id = str(uuid.uuid5(uuid.UUID(doc_id), str(idx)))
            payload = {
                "doc_id": doc_id,
                "chunk_id": idx,
                "title": Path(pdf_path).name,
                "path": str(pdf_path),
                "snippet": snippet
            }
            points.append(PointStruct(id=point_id, vector=vector.tolist(), payload=payload))
        client.upsert(collection_name=COLLECTION_NAME, points=points)


if __name__ == "__main__":
    process_and_index()
    print("Indexacion completa en Qdrant")
