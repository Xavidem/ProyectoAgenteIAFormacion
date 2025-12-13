import os
import time
import json
import logging
import subprocess
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

# ============================
#  INICIO: Configuración básica
# ============================
METADATA_PATH = Path(os.getenv("METADATA_PATH", "metadata_master_1.json")).resolve()
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "docs")
SP_WATCH_INTERVAL = int(os.getenv("SP_WATCH_INTERVAL", "60"))

# ============================
#  FIN: Configuración básica
# ============================

# ============================
#  INICIO: Funciones auxiliares
# ============================

def run_metadata_extractor():
    logging.info("[watcher] Ejecutando metadata_extractor.py…")
    cmd = ["python", "metadata_extractor.py", METADATA_PATH.name]
    subprocess.run(cmd, check=True)
    logging.info("[watcher] metadata_extractor.py finalizado correctamente")


def run_text_extractor():
    logging.info("[watcher] Ejecutando text_extractor.py…")
    cmd = ["python", "text_extractor.py"]
    subprocess.run(cmd, check=True)
    logging.info("[watcher] text_extractor.py finalizado correctamente")


def load_metadata_ids() -> set[str]:
    if not METADATA_PATH.exists():
        logging.warning("[watcher] METADATA_PATH no existe todavía: %s", METADATA_PATH)
        return set()

    try:
        records = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        logging.error("[watcher] Error leyendo %s: %s", METADATA_PATH, e)
        return set()

    ids: set[str] = set()
    if isinstance(records, list):
        for r in records:
            doc_id = r.get("id")
            if doc_id:
                ids.add(str(doc_id))
    logging.info("[watcher] Metadatos cargados: %d doc_ids", len(ids))
    return ids

def collect_qdrant_doc_ids(client: QdrantClient) -> set[str]:
    doc_ids: set[str] = set()
    offset = None

    while True:
        points, offset = client.scroll(
            collection_name=QDRANT_COLLECTION,
            limit=256,
            with_payload=True,
            offset=offset,
        )
        if not points:
            break

        for p in points:
            payload = p.payload or {}
            did = payload.get("doc_id")
            if did is not None:
                doc_ids.add(str(did))

        if offset is None:
            break

    logging.info("[watcher] Qdrant contiene %d doc_ids", len(doc_ids))
    return doc_ids

def delete_orphan_docs(client: QdrantClient, valid_ids: set[str]):
    current_ids = collect_qdrant_doc_ids(client)
    orphans = sorted(current_ids - valid_ids)

    if not orphans:
        logging.info("[watcher] No hay doc_ids huérfanos que borrar en Qdrant")
        return

    logging.info("[watcher] Eliminando %d doc_ids huérfanos en Qdrant…", len(orphans))
    for doc_id in orphans:
        logging.info("[watcher] Borrando doc_id=%s", doc_id)
        client.delete(
            collection_name=QDRANT_COLLECTION,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="doc_id",
                        match=MatchValue(value=doc_id),
                    )
                ]
            ),
        )

def sync_once():
    run_metadata_extractor()
    run_text_extractor()

    valid_ids = load_metadata_ids()
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    delete_orphan_docs(client, valid_ids)


# ============================
#  FIN: Funciones auxiliares
# ============================

# ============================
#  INICIO: Main loop
# ============================

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logging.info(
        "[watcher] Iniciado. Sincronizando cada %d segundos. "
        "Colección Qdrant=%s, metadata=%s",
        SP_WATCH_INTERVAL,
        QDRANT_COLLECTION,
        METADATA_PATH,
    )

    while True:
        try:
            sync_once()
        except subprocess.CalledProcessError as e:
            logging.error("[watcher] Error ejecutando un subproceso: %s", e)
        except Exception as e:
            logging.exception("[watcher] Error inesperado en el ciclo: %s", e)

        logging.info("[watcher] Dormimos %d segundos antes del siguiente ciclo…", SP_WATCH_INTERVAL)
        time.sleep(SP_WATCH_INTERVAL)


if __name__ == "__main__":
    main()
