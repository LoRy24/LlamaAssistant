#!/usr/bin/env bash
# ============================================================
#  RaspAI - Script di installazione
# ============================================================
#  Installa e compila tutto il necessario DENTRO la cartella
#  del repository, in modo che il progetto sia autocontenuto:
#
#    - dipendenze di sistema (apt)
#    - llama.cpp        (clone + build)
#    - whisper.cpp      (clone + build + modello base)
#    - virtualenv Python + pacchetti (incluso Piper TTS)
#    - voce italiana per Piper
#    - modello AI Qwen (.gguf)
#
#  Uso:   ./setup.sh
# ============================================================

set -e  # esci al primo errore

# ----- Cartella del repository (dove si trova questo script) -----
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

# ----- Colori per i messaggi -----
C_OK="\033[1;32m"; C_INFO="\033[1;36m"; C_WARN="\033[1;33m"; C_END="\033[0m"
say()  { echo -e "${C_INFO}==>${C_END} $*"; }
ok()   { echo -e "${C_OK}  ok${C_END} $*"; }
warn() { echo -e "${C_WARN}  !!${C_END} $*"; }

# Numero di core per la compilazione
JOBS="$(nproc 2>/dev/null || echo 2)"

echo "============================================="
echo "   RaspAI  -  installazione"
echo "   cartella: $REPO"
echo "   core per la build: $JOBS"
echo "============================================="

# ============================================================
# 1) DIPENDENZE DI SISTEMA
# ============================================================
say "Installazione pacchetti di sistema (richiede sudo)..."
sudo apt-get update
sudo apt-get install -y \
    build-essential cmake git wget \
    python3 python3-pip python3-venv python3-tk \
    espeak-ng alsa-utils \
    libopenblas-dev
ok "pacchetti di sistema installati"

# ============================================================
# 2) LLAMA.CPP  -  clone + build
# ============================================================
if [ ! -d "$REPO/llama.cpp" ]; then
    say "Clono llama.cpp..."
    git clone https://github.com/ggml-org/llama.cpp.git "$REPO/llama.cpp"
else
    say "llama.cpp gia' presente, aggiorno..."
    git -C "$REPO/llama.cpp" pull --ff-only || warn "pull saltato"
fi

say "Compilo llama.cpp (puo' richiedere parecchi minuti)..."
cmake -S "$REPO/llama.cpp" -B "$REPO/llama.cpp/build" \
      -DCMAKE_BUILD_TYPE=Release
cmake --build "$REPO/llama.cpp/build" --config Release -j "$JOBS"

# verifica i binari attesi dallo script Python
if [ -x "$REPO/llama.cpp/build/bin/llama-server" ]; then
    ok "llama-server compilato"
else
    warn "llama-server non trovato dopo la build - controlla l'output sopra"
fi

# ============================================================
# 3) WHISPER.CPP  -  clone + build + modello
# ============================================================
if [ ! -d "$REPO/whisper.cpp" ]; then
    say "Clono whisper.cpp..."
    git clone https://github.com/ggml-org/whisper.cpp.git "$REPO/whisper.cpp"
else
    say "whisper.cpp gia' presente, aggiorno..."
    git -C "$REPO/whisper.cpp" pull --ff-only || warn "pull saltato"
fi

say "Compilo whisper.cpp..."
cmake -S "$REPO/whisper.cpp" -B "$REPO/whisper.cpp/build" \
      -DCMAKE_BUILD_TYPE=Release
cmake --build "$REPO/whisper.cpp/build" --config Release -j "$JOBS"

if [ -x "$REPO/whisper.cpp/build/bin/whisper-cli" ]; then
    ok "whisper-cli compilato"
else
    warn "whisper-cli non trovato dopo la build - controlla l'output sopra"
fi

# modello whisper "base" (lo script Python usa ggml-base.bin)
WMODEL="$REPO/whisper.cpp/models/ggml-base.bin"
if [ ! -f "$WMODEL" ]; then
    say "Scarico il modello whisper 'base'..."
    bash "$REPO/whisper.cpp/models/download-ggml-model.sh" base
else
    ok "modello whisper gia' presente"
fi

# ============================================================
# 4) AMBIENTE PYTHON (venv) + PACCHETTI
# ============================================================
if [ ! -d "$REPO/venv" ]; then
    say "Creo l'ambiente virtuale Python (venv)..."
    python3 -m venv "$REPO/venv"
else
    say "venv gia' presente"
fi

say "Installo i pacchetti Python nel venv..."
# nota: tkinter NON si installa con pip, e' il pacchetto di sistema python3-tk
"$REPO/venv/bin/pip" install --upgrade pip
if [ -f "$REPO/requirements.txt" ]; then
    "$REPO/venv/bin/pip" install -r "$REPO/requirements.txt"
else
    "$REPO/venv/bin/pip" install piper-tts
fi
ok "pacchetti Python installati"

# ============================================================
# 5) VOCE PIPER (italiano, femminile)
# ============================================================
PIPER_DIR="$REPO/piper"
mkdir -p "$PIPER_DIR"
VOICE_BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main/it/it_IT/paola/medium"
VOICE_ONNX="$PIPER_DIR/it_IT-paola-medium.onnx"
VOICE_JSON="$PIPER_DIR/it_IT-paola-medium.onnx.json"

if [ ! -f "$VOICE_ONNX" ]; then
    say "Scarico la voce italiana per Piper..."
    wget -q --show-progress -O "$VOICE_ONNX" "$VOICE_BASE/it_IT-paola-medium.onnx"
    wget -q --show-progress -O "$VOICE_JSON" "$VOICE_BASE/it_IT-paola-medium.onnx.json"
    ok "voce Piper scaricata"
else
    ok "voce Piper gia' presente"
fi

# ============================================================
# 6) MODELLO AI  (Qwen2.5 1.5B, quantizzato Q4_K_M ~1 GB)
# ============================================================
mkdir -p "$REPO/models"
QWEN="$REPO/models/qwen1.5b.gguf"
QWEN_URL="https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf"

if [ ! -f "$QWEN" ]; then
    say "Scarico il modello Qwen2.5 1.5B (~1 GB, puo' volerci un po')..."
    wget -q --show-progress -O "$QWEN" "$QWEN_URL"
    ok "modello Qwen scaricato"
else
    ok "modello Qwen gia' presente"
fi

# ============================================================
# FINE
# ============================================================
chmod +x "$REPO/run.sh" 2>/dev/null || true

echo
echo -e "${C_OK}=============================================${C_END}"
echo -e "${C_OK}   Installazione completata!${C_END}"
echo -e "${C_OK}=============================================${C_END}"
echo
echo "  Per avviare RaspAI:"
echo "      ./run.sh"
echo
