import os
import json
import time
import hashlib
import uuid
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent
from text_extractor import index_single_pdf, delete_by_doc_id

WATCH_DIR = Path(os.getenv("WATCH_DIR", "data/raw/proyecto-chatbot-materiales"))
METADATA_PATH = Path(os.getenv("METADATA_PATH", "metadata_master.json"))

# Creamos los metadatos si no existen
if not METADATA_PATH.exists():
    METADATA_PATH.write_text("[]", encoding="utf-8")

def load_metadata_by_path():
    recs = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    by_path = {r["path"]: r for r in recs if "path" in r}
    return by_path

def save_metadata(by_path):
    recs = list(by_path.values())
    tmp = METADATA_PATH.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(recs, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(METADATA_PATH)

# Vamos a ver si tenemos metadata para reusar, sino generamos ID desde el Path
"""DESCRIPCION PRINCIPAL DE LA FUNCION: Sirve para generar un ID unico"""
def mk_doc_id(pdf_path: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, pdf_path))

"""DESCRIPCION PRINCIPAL DE LA FUNCION: Sirve para generar una firma unica de un archivo"""
def file_sig(p: Path) -> str:
    try:
        st = p.stat()
        return f"{int(st.st_mtime)}:{st:st_size}"
    except FileNotFoundError:
        return ""

"""DESCRIPCION DE CLASE: Esta clase implementa un debounce para evitar eventos duplicados"""
class Debounce:
    def __init__(self, seconds=1.5):
        self.seconds = seconds
        self.last = {}
    def allow(self, key):
        now = time.time()
        t = self.last.get(key, 0)
        if now - t >= self.seconds:
            self.last[key] = now
            return True
        return False
    
debounce = Debounce(1.5)

def process_upsert(path: Path):
    by_path = load_metadata_by_path()
    old = by_path.get(str(path))
    new_sig = file_sig(path)
    if old and old.get("sig") == new_sig:
        return
    rec = by_path.get(str(path))
    if rec is None:
        rec = {
            "id": mk_doc_id(str(path)),
            "path": str(path),
            "type": "pdf",
            "title": path.name,
            "sig": file_sig(path)
        }
        by_path[str(path)] = rec
    else:
        rec["sig"] = file_sig(path) # actualizamos la firma

    print(f"[watcher] upsert -> {path}")
    index_single_pdf(rec["id"], str(path))
    save_metadata(by_path)

def process_delete(path: Path):
    by_path = load_metadata_by_path()
    rec = by_path.get(str(path))
    if rec:
        print(f"[watcher] delete -> {path}")
        delete_by_doc_id(rec["id"])
        save_metadata(by_path)
    else:
        print(f"[watcher] delete (sin metadata) -> {path}")

"""DESCRIPCION DE CLASE: Esta clase maneja los eventos del sistema de archivos de PDFs"""
class PDFHandler(FileSystemEventHandler):
    def on_any_event(self, event: FileSystemEvent):
        if event.is_directory:
            return
        p = Path(event.src_path)
        if p.suffix.lower() != ".pdf":
            return
        
        # Hacemos debounce para evitar duplicados
        key = (event.event_type, str(p))
        if not debounce.allow(key):
            return
        
        if event.event_type in ("created", "modified"):
            target = Path(getattr(event, "dest_path", p))
            if target.exists():
                process_upsert(target)
        elif event.event_type == "moved":
            src = Path(event.src_path)
            dst = Path(event.dest_path)
            by_path = load_metadata_by_path()
            rec = by_path.pop(str(src), None)
            if rec:
                rec["path"] = str(dst)
                rec["sig"] = file_sig(dst)
                by_path[str(dst)] = rec
                save_metadata(by_path)
                process_upsert(dst)
        elif event.event_type == "deleted":
            process_delete(p)


"""DESCRIPCION PRINCIPAL DE LA FUNCION: Se hace un escaneo, indexa nuevos o modificados y elimina metadatos inexistentes"""
def initial_resync():
    by_path = load_metadata_by_path()
    # Hacemos upsert de todos los PDFs presentes
    present = set()
    for p in WATCH_DIR.rglob("*.pdf"):
        present.add(str(p))
        if str(p) not in by_path or by_path[str(p)].get("sig") != file_sig(p):
            process_upsert(p)

    
    removed = [k for k in by_path.keys() if k not in present]
    for k in removed:
        process_delete(Path(k))


def main():
    WATCH_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[watcher] Observando directorio: {WATCH_DIR.resolve()}")
    initial_resync()

    observer = Observer()
    observer.schedule(PDFHandler(), str(WATCH_DIR), recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(1.0)
    finally:
        observer.stop()
        observer.join()

if __name__ == "__main__":
    main()