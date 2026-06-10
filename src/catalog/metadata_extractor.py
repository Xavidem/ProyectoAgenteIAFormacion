from __future__ import annotations
import json
import uuid
import csv
import hashlib
import os
from pathlib import Path
from typing import List, Dict, Optional

SUPPORTED_EXTS = {"pdf", "docx"}

DEFAULT_OUTPUT = "metadata_master_1.json"
DATA_ROOT = Path(os.getenv("DATA_ROOT", "/app/data/pdfs"))

# Namespace estable para derivar UUIDs deterministas via uuid5(NS, sha256).
# Cambiarlo invalida todos los doc_id ya indexados.
DOC_NAMESPACE = uuid.UUID("9b6c1a0e-1b6a-4d6c-9c1a-0e1b6a4d6c9c")


def _ext(name: str) -> str:
    return name.rsplit(".", 1)[-1].lower() if "." in name else ""


def _sha256_of_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for blk in iter(lambda: f.read(chunk_size), b""):
            h.update(blk)
    return h.hexdigest()


def _doc_id_from_hash(sha256_hex: str) -> str:
    return str(uuid.uuid5(DOC_NAMESPACE, sha256_hex))


def _load_previous(output_path: Path) -> Dict[str, Dict]:
    if not output_path.exists() or output_path.stat().st_size == 0:
        return {}
    try:
        data = json.loads(output_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, list):
        return {}
    return {rec["id"]: rec for rec in data if isinstance(rec, dict) and rec.get("id")}


def collect_metadata_local(previous: Optional[Dict[str, Dict]] = None) -> List[Dict]:
    """Recorre DATA_ROOT, calcula sha256+doc_id deterministas y conserva los
    campos enriquecidos (author, created, indexed_at, ...) del JSON anterior
    cuando el contenido del fichero no ha cambiado.
    """
    previous = previous or {}
    records: List[Dict] = []
    seen_exts = set()

    for path in DATA_ROOT.rglob("*"):
        if not path.is_file():
            continue

        ext = _ext(path.name)
        seen_exts.add(ext)

        if ext not in SUPPORTED_EXTS:
            continue

        rel_path = path.relative_to(DATA_ROOT).as_posix()
        size_bytes = path.stat().st_size
        sha256 = _sha256_of_file(path)
        doc_id = _doc_id_from_hash(sha256)

        prev = previous.get(doc_id) or {}
        prev_extra = prev.get("extra") or {}

        rec: Dict = {
            "id": doc_id,
            "path": rel_path,
            "type": ext,
            "title": prev.get("title") or path.stem,
            "author": prev.get("author", ""),
            "created": prev.get("created", ""),
            "modified": prev.get("modified", ""),
            "language": prev.get("language", ""),
            "extra": {
                "local_rel_path": rel_path,
                "size_bytes": size_bytes,
                "sha256": sha256,
                "indexed_at": prev_extra.get("indexed_at", ""),
                "chunks_count": prev_extra.get("chunks_count", 0),
                "indexed_sha256": prev_extra.get("indexed_sha256", ""),
            },
        }
        records.append(rec)

    print(f"[metadata] Extensiones vistas: {sorted(seen_exts)}")
    print(f"[metadata] Ficheros listados localmente: {len(records)}")
    return records


def save_records(records: List[Dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".csv":
        keys = ["id", "path", "type", "title", "author", "created", "modified", "language"]
        with output_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in records:
                w.writerow({k: r.get(k, "") for k in keys})
    else:
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Genera/actualiza metadata_master_1.json recorriendo DATA_ROOT"
    )
    parser.add_argument(
        "output",
        nargs="?",
        default=DEFAULT_OUTPUT,
        help="Salida (.json o .csv). Por defecto metadata_master_1.json",
    )
    args = parser.parse_args()

    out_path = Path(args.output).expanduser().resolve()
    previous = _load_previous(out_path)

    print(f"[metadata] Iniciando recopilación LOCAL desde {DATA_ROOT}…")
    if previous:
        print(f"[metadata] JSON previo cargado: {len(previous)} registros conservados como base")
    recs = collect_metadata_local(previous=previous)
    print(f"[metadata] Metadatos obtenidos: {len(recs)}")

    save_records(recs, out_path)
    print(f"[metadata] Guardado en: {out_path}")


if __name__ == "__main__":
    main()
