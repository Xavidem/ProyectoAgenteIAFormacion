from __future__ import annotations
import json
import uuid
import csv
from pathlib import Path
from typing import List, Dict
from sharepoint_fetcher import list_all_files_under_docroot, build_canonical_url

# Extensiones soportadas
SUPPORTED_EXTS = {"pdf", "docx", "pptx"}

DEFAULT_OUTPUT = "metadata_master_1.json"

def _ext(name: str) -> str:
    return (name.rsplit(".", 1)[-1].lower() if "." in name else "")

# ============================================================
#  INICIO: RECOLECCION DE SHAREPOINT
# ============================================================

def collect_metadata_sharepoint() -> List[Dict]:
    records: List[Dict] = []
    n = 0
    seen_exts = set()

    for item in list_all_files_under_docroot():
        name = item.get("Name", "")
        ext = _ext(name)
        seen_exts.add(ext)


        if ext not in SUPPORTED_EXTS:
            continue

        server_rel = item.get("ServerRelativeUrl", "")
        url_click = build_canonical_url(server_rel)
        modified = item.get("TimeLastModified", "")
        unique_id = item.get("UniqueId", "")

        rec = {
            "id": str(uuid.uuid4()),
            "path": url_click,
            "type": ext,
            "title": name.rsplit(".", 1)[0],
            "author": item.get("Author", {}).get("Title", ""),
            "created": "",
            "modified": modified,
            "extra": {
                "unique_id": unique_id,
                "server_relative_url": server_rel,
                "length_bytes": item.get("Length", ""),
            }
        }
        records.append(rec)
        n += 1
    print(f"[metadata] Extensiones vistas: {sorted(seen_exts)}")
    print(f"[metadata] Ficheros listados en SP: {n}")
    return records

# ============================================================
#  FIN: RECOLECCION DE SHAREPOINT
# ============================================================

# ============================================================
#  INICIO: GUARDADO DE REGISTROS
# ============================================================
def save_records(records: List[Dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".csv":
        keys = ["id","path","type","title","author","created","modified"]
        with output_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in records:
                w.writerow({k: r.get(k,"") for k in keys})
    else:
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Genera metadata_master.json desde SharePoint")
    parser.add_argument("output", nargs="?", default=DEFAULT_OUTPUT, help="Salida (.json o .csv). Por defecto metadata_master.json")
    args = parser.parse_args()

    print("[metadata] Iniciando recopilación desde SharePoint…")
    recs = collect_metadata_sharepoint()
    print(f"[metadata] Metadatos obtenidos: {len(recs)}")
    out_path = Path(args.output).expanduser().resolve()
    save_records(recs, out_path)
    print(f"[metadata] Guardado en: {out_path}")

if __name__ == "__main__":
    main()