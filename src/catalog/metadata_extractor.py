import os
import csv
import json
import uuid
from pathlib import Path
from datetime import datetime
from docx import Document
from pptx import Presentation
import argparse

try:
    import fitz
except ImportError:
    raise ImportError("Por favor instala PyMuPDF: pip install pymupdf")

#Constantes necesarias para los formatos soportados
SUPPORTED_VIDEO = {"mp4"}
SUPPORTED_PDF = {"pdf"}

#Funcion principal de extraccion de metadatos de pdf
"""DESCRIPCION PRINCIPAL DE LA FUNCION:
Extrae metadatos de un PDF usando PyMuPDF
OUT: Devuelve un diccionario con el titulo, autor, fechas y numero de paginas del PDF
"""
def extract_pdf_metadata(file_path: Path):
    doc = fitz.open(file_path)
    metadata = doc.metadata
    pages = doc.page_count
    doc.close()
    return{
        "title": metadata.get("title") or file_path.stem,
        "author": metadata.get("author") or "",
        "creation_date": metadata.get("creationDate") or "",
        "modified_date": metadata.get("modDate") or "",
        "pages": pages 
    }

#Funcion principal de extraccion de metadatos de pdf
"""DESCRIPCION PRINCIPAL DE LA FUNCION:
Extrae metadatos de un DOCX usando PyMuPDF
OUT: Devuelve un diccionario con el titulo, autor, fechas y numero de paginas del DOCX
"""
def extract_docx_metadata(file_path: Path):
    doc = Document(file_path)
    core_properties = doc.core_properties
    doc.close()
    return{
        "title": core_properties.title or file_path.stem,
        "author": core_properties.author or "",
        "creation_date": core_properties.created.isoformat() if core_properties.created else "",
        "modified_date": core_properties.modified.isoformat() if core_properties.modified else "",
        "pages": len(doc.element.body.findall(".//w:sectPr"))
    }

#Funcion principal de extraccion de metadatos de pdf
"""DESCRIPCION PRINCIPAL DE LA FUNCION:
Extrae metadatos de un PPTX usando PyMuPDF
OUT: Devuelve un diccionario con el titulo, autor, fechas y numero de paginas del PPTX
"""
def extract_pptx_metadata(file_path: Path):
    presentation = Presentation(file_path)
    core_properties = presentation.core_properties
    presentation.close()
    return{
        "title": core_properties.title or file_path.stem,
        "author": core_properties.author or "",
        "creation_date": core_properties.created.isoformat() if core_properties.created else "",
        "modified_date": core_properties.modified.isoformat() if core_properties.modified else "",
        "pages": len(presentation.slides)
    }
 
#Funcion principal de extraccion de metadatos de video
"""DESCRIPCION PRINCIPAL DE LA FUNCION:
Extrae metadatos de un video usando ffprobe
OUT: Devuelve un diccionario con la duracion, el tamaño y el codec
"""
def extract_video_metadate(file_path: Path):
    pass


#Funcion que recolecta los metadatos
"""DESCRIPCION PRINCIPAL DE LA FUNCION:
Recorre recursivamente el directorio raiz y extrae metadatos de los ficheros
OUT: Devuelve una lista de registros 
"""
def collect_metadata(root_dir: Path):
    records = []
    for path in root_dir.rglob("*"):
        if not path.is_file():
            continue
        ext = path.suffix.lower().lstrip(".")
        record = {
            "id": str(uuid.uuid4()),
            "path": str(path.resolve()),
            "type": ext,
            "title": "",
            "author": "",
            "created": "",
            "modified": "",
            "extra": {}
        }

        #Fechas del sistema de ficheros
        stat = path.stat()
        birth_ts = getattr(stat, "st_birthtime", None)
        if birth_ts:
            record["created"] = datetime.fromtimestamp(birth_ts).isoformat()
        record["modified"] = datetime.fromtimestamp(stat.st_mtime).isoformat()

        if ext in SUPPORTED_PDF:
            md = extract_pdf_metadata(path)
            if md["creation_date"]:
                record["created"] = md["creation_date"]
            
            if md["modified_date"]:
                record["modified"] = md["modified_date"]
            
            record.update({
                "title": md["title"],
                "author": md["author"],
                "extra": {"pages": md["pages"]}
            })
        elif ext in SUPPORTED_VIDEO:
            continue
        else:
            #Ignoramos archivos no soportados por el formato
            continue

        records.append(record)
    return records

#Funcion que guarda la lista de registros
"""DESCRIPCION PRINCIPAL DE LA FUNCION:
Guarda la lista de records en formato JSON/CSV
OUT: Devuelve una fichero JSON/CSV
"""
def save_records(records, output_path: Path):
    if output_path.suffix.lower() == ".json":
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent= 2)
    elif output_path.suffix.lower() == ".csv":
        fieldnames = ["id", "path", "type", "title", "author", "created", "modified"]
        extra_keys = set().union(*(rec.get("extra", {}).keys() for rec in records))
        fieldnames += sorted(extra_keys)
        with output_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rec in records:
                row = {k: rec.get(k, "") for k in ["id", "path", "type", "title", "author", "created", "modified"]}
                for key in extra_keys:
                    row[key] = rec.get("extra", {}).get(key, "")
                writer.writerow(row)
    else:
        raise ValueError("El archivo de salida debe ser formato .json o .csv")
    


# FUNCION MAIN PRINCIPAL
def main():
    parser = argparse.ArgumentParser(description="Recolecta metadatos")
    parser.add_argument("root_dir", type=Path, help="Directorio raiz con los ficheros PDF")
    parser.add_argument("output_file", type=Path, help="Archivo de salida con formato .json o .csv")
    args = parser.parse_args()

    print(f"Iniciando recopilacion desde {args.root_dir}")
    records = collect_metadata(args.root_dir)
    print(f"Metadatos obtenidos para {len(records)} archivos")
    save_records(records, args.output_file)
    print(f"Metadatos guardados en {args.output_file.resolve()}")

if __name__ == "__main__":
    main()