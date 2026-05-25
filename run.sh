#!/usr/bin/env bash
# Cartella del repository (dove si trova questo script)
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

PY="$REPO/venv/bin/python3"
APP="$REPO/voice_assistant.py"

# --- Controlli minimi ---
if [ ! -x "$PY" ]; then
    echo "ERRORE: ambiente virtuale non trovato."
    echo "Esegui prima l'installazione:  ./setup.sh"
    exit 1
fi

if [ ! -f "$APP" ]; then
    echo "ERRORE: voice_assistant.py non trovato in $REPO"
    exit 1
fi

# --- Avvio ---
echo "Avvio di RaspAI..."
exec "$PY" "$APP"
