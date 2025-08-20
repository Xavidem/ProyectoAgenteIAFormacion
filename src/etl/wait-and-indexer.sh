#!/bin/sh
set -e

echo "Esperando a que Qdrant este listo..."
sleep 7
echo "Qdrant listo. Iniciando indexacion..."

python /app/text_extractor.py

echo "Indexacion completada"