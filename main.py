#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Assistente vocale offline per Raspberry Pi
==========================================
Parla con un modello AI (Qwen 1.5B) gestito da llama.cpp.

Catena: microfono -> whisper.cpp (speech-to-text) -> llama-server (AI) -> espeak-ng (text-to-speech)

Dipendenze di sistema (installabili con apt):
    sudo apt install espeak-ng alsa-utils python3-tk
    (arecord fa parte di alsa-utils, gia' presente su Raspberry Pi OS)

llama-server deve essere compilato insieme a llama.cpp:
    si trova di solito in llama.cpp/build/bin/llama-server
"""

import os
import sys
import json
import time
import signal
import threading
import subprocess
import tempfile
import urllib.request
import urllib.error
import tkinter as tk
from tkinter import font as tkfont

# ============================================================
# CONFIGURAZIONE  -  modifica qui i percorsi se necessario
# ============================================================
BASE          = "/home/pi"
WHISPER       = f"{BASE}/whisper.cpp/build/bin/whisper-cli"
WHISPER_MODEL = f"{BASE}/whisper.cpp/models/ggml-base.bin"

LLAMA_SERVER  = f"{BASE}/llama.cpp/build/bin/llama-server"
LLAMA_MODEL   = f"{BASE}/llama.cpp/models/qwen1.5b.gguf"

# Server llama.cpp
LLAMA_HOST    = "127.0.0.1"
LLAMA_PORT    = 8080
LLAMA_THREADS = 4          # core del Raspberry da usare
LLAMA_CTX     = 2048       # dimensione del contesto

# Audio
ARECORD_DEVICE = "default" # "default" oppure es. "plughw:1,0" -- vedi `arecord -l`
WHISPER_LANG   = "it"      # lingua del parlato ("it", "en", "auto", ...)

# Text-to-speech (espeak-ng)
TTS_VOICE = "it"           # voce espeak-ng
TTS_SPEED = 160            # parole al minuto

# Prompt di sistema dato al modello
SYSTEM_PROMPT = (
    "Sei un assistente vocale gentile e conciso. "
    "Rispondi in italiano in modo chiaro e breve, adatto alla lettura ad alta voce."
)

# ============================================================
# PALETTE / TEMA GRAFICO
# ============================================================
COL_BG      = "#0d1117"   # sfondo principale (blu notte)
COL_PANEL   = "#161b22"   # pannelli
COL_CARD    = "#1c2330"   # card della risposta
COL_ACCENT  = "#5eead4"   # accento (verde acqua)
COL_ACCENT2 = "#f472b6"   # accento secondario (rosa)
COL_TEXT    = "#e6edf3"   # testo principale
COL_MUTE    = "#7d8590"   # testo secondario
COL_DANGER  = "#f85149"   # rosso (chiudi / errori)
COL_OK      = "#3fb950"   # verde (ok)


# ============================================================
# GESTIONE DEL SERVER LLAMA
# ============================================================
class LlamaServer:
    """Avvia e mantiene caldo llama-server in background."""

    def __init__(self):
        self.proc = None
        self.url_completion = f"http://{LLAMA_HOST}:{LLAMA_PORT}/completion"
        self.url_health     = f"http://{LLAMA_HOST}:{LLAMA_PORT}/health"

    def is_alive(self):
        """True se il server risponde all'endpoint /health."""
        try:
            with urllib.request.urlopen(self.url_health, timeout=2) as r:
                data = json.loads(r.read().decode("utf-8"))
                return data.get("status") in ("ok", "no slot available")
        except Exception:
            return False

    def start(self):
        """Avvia il server se non e' gia' attivo."""
        if self.is_alive():
            return  # qualcuno l'ha gia' avviato, lo riusiamo

        if not os.path.exists(LLAMA_SERVER):
            raise FileNotFoundError(
                f"llama-server non trovato in:\n{LLAMA_SERVER}\n"
                "Compila llama.cpp con il target server."
            )
        if not os.path.exists(LLAMA_MODEL):
            raise FileNotFoundError(f"Modello non trovato:\n{LLAMA_MODEL}")

        cmd = [
            LLAMA_SERVER,
            "-m", LLAMA_MODEL,
            "--host", LLAMA_HOST,
            "--port", str(LLAMA_PORT),
            "-t", str(LLAMA_THREADS),
            "-c", str(LLAMA_CTX),
        ]
        # Avvia in un nuovo process group: cosi' lo possiamo chiudere insieme ai figli
        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid,
        )

    def wait_ready(self, timeout=120):
        """Attende che il server sia pronto a rispondere."""
        start = time.time()
        while time.time() - start < timeout:
            if self.is_alive():
                return True
            if self.proc and self.proc.poll() is not None:
                raise RuntimeError("llama-server si e' chiuso inaspettatamente.")
            time.sleep(1)
        return False

    def ask(self, user_text):
        """Invia un prompt al modello e restituisce la risposta come stringa."""
        prompt = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{user_text}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        payload = json.dumps({
            "prompt": prompt,
            "n_predict": 320,
            "temperature": 0.7,
            "top_p": 0.9,
            "stop": ["<|im_end|>", "<|im_start|>"],
        }).encode("utf-8")

        req = urllib.request.Request(
            self.url_completion,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=180) as r:
            data = json.loads(r.read().decode("utf-8"))
        return data.get("content", "").strip()

    def stop(self):
        """Termina il server e tutti i suoi processi figli."""
        if self.proc and self.proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
                self.proc.wait(timeout=5)
            except Exception:
                try:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                except Exception:
                    pass


# ============================================================
# APPLICAZIONE GUI
# ============================================================
class VoiceAssistantApp:
    RECORD_SECONDS = 8  # durata massima registrazione

    def __init__(self, root):
        self.root = root
        self.llama = LlamaServer()
        self.busy = False
        self.last_answer = ""
        self.rec_proc = None

        self._build_ui()
        # Avvia il server in un thread separato per non bloccare la finestra
        threading.Thread(target=self._boot_server, daemon=True).start()

    # ---------- COSTRUZIONE INTERFACCIA ----------
    def _build_ui(self):
        self.root.title("Assistente Vocale")
        self.root.configure(bg=COL_BG)
        self.root.attributes("-fullscreen", True)
        # ESC come scorciatoia d'emergenza per uscire
        self.root.bind("<Escape>", lambda e: self.quit_app())

        # --- Font ---
        self.f_title  = tkfont.Font(family="DejaVu Sans", size=30, weight="bold")
        self.f_sub    = tkfont.Font(family="DejaVu Sans", size=12)
        self.f_body   = tkfont.Font(family="DejaVu Serif", size=17)
        self.f_btn    = tkfont.Font(family="DejaVu Sans", size=18, weight="bold")
        self.f_small  = tkfont.Font(family="DejaVu Sans", size=11)
        self.f_status = tkfont.Font(family="DejaVu Sans Mono", size=11)

        # --- Barra superiore ---
        topbar = tk.Frame(self.root, bg=COL_BG)
        topbar.pack(fill="x", padx=36, pady=(28, 8))

        title_wrap = tk.Frame(topbar, bg=COL_BG)
        title_wrap.pack(side="left")
        tk.Label(title_wrap, text="◈  ORACOLO", font=self.f_title,
                 fg=COL_TEXT, bg=COL_BG).pack(anchor="w")
        tk.Label(title_wrap, text="assistente vocale offline · qwen 1.5b",
                 font=self.f_sub, fg=COL_MUTE, bg=COL_BG).pack(anchor="w")

        # Pulsante CHIUDI (task-kill)
        self.btn_close = tk.Button(
            topbar, text="✕  CHIUDI", font=self.f_btn,
            fg="#ffffff", bg=COL_DANGER, activebackground="#ff6b63",
            activeforeground="#ffffff", relief="flat", bd=0,
            padx=22, pady=10, cursor="hand2",
            command=self.quit_app,
        )
        self.btn_close.pack(side="right")

        # Linea accento sotto la barra
        tk.Frame(self.root, bg=COL_ACCENT, height=2).pack(fill="x", padx=36)

        # --- Area centrale: la risposta ---
        center = tk.Frame(self.root, bg=COL_BG)
        center.pack(fill="both", expand=True, padx=36, pady=24)

        card = tk.Frame(center, bg=COL_CARD, highlightthickness=1,
                        highlightbackground="#2a3340")
        card.pack(fill="both", expand=True)

        cap = tk.Frame(card, bg=COL_CARD)
        cap.pack(fill="x", padx=26, pady=(20, 6))
        tk.Label(cap, text="RISPOSTA", font=self.f_small,
                 fg=COL_ACCENT, bg=COL_CARD).pack(side="left")
        self.lbl_heard = tk.Label(cap, text="", font=self.f_small,
                                  fg=COL_MUTE, bg=COL_CARD)
        self.lbl_heard.pack(side="right")

        # Casella di testo con scrollbar per la risposta
        txt_wrap = tk.Frame(card, bg=COL_CARD)
        txt_wrap.pack(fill="both", expand=True, padx=26, pady=(0, 20))

        scroll = tk.Scrollbar(txt_wrap)
        scroll.pack(side="right", fill="y")

        self.txt = tk.Text(
            txt_wrap, font=self.f_body, fg=COL_TEXT, bg=COL_CARD,
            wrap="word", relief="flat", bd=0, padx=4, pady=4,
            insertbackground=COL_TEXT, yscrollcommand=scroll.set,
            highlightthickness=0,
        )
        self.txt.pack(side="left", fill="both", expand=True)
        scroll.config(command=self.txt.yview)
        self._set_text("Premi  «PARLA»  e fai la tua domanda.\n\n"
                       "Avvio del modello in corso...", muted=True)
        self.txt.config(state="disabled")

        # --- Barra inferiore: pulsanti azione ---
        bottom = tk.Frame(self.root, bg=COL_BG)
        bottom.pack(fill="x", padx=36, pady=(0, 22))

        self.btn_talk = tk.Button(
            bottom, text="🎙   PARLA", font=self.f_btn,
            fg=COL_BG, bg=COL_ACCENT, activebackground="#7ff0dd",
            activeforeground=COL_BG, relief="flat", bd=0,
            padx=40, pady=20, cursor="hand2",
            command=self.on_talk,
        )
        self.btn_talk.pack(side="left", expand=True, fill="x", padx=(0, 10))

        self.btn_listen = tk.Button(
            bottom, text="🔊   ASCOLTA", font=self.f_btn,
            fg=COL_TEXT, bg=COL_PANEL, activebackground="#222b38",
            activeforeground=COL_TEXT, relief="flat", bd=0,
            padx=40, pady=20, cursor="hand2",
            state="disabled", command=self.on_listen,
        )
        self.btn_listen.pack(side="left", expand=True, fill="x", padx=(10, 0))

        # --- Barra di stato ---
        statusbar = tk.Frame(self.root, bg=COL_PANEL)
        statusbar.pack(fill="x", side="bottom")
        self.dot = tk.Label(statusbar, text="●", font=self.f_status,
                            fg=COL_DANGER, bg=COL_PANEL)
        self.dot.pack(side="left", padx=(14, 6), pady=6)
        self.lbl_status = tk.Label(statusbar, text="Avvio del modello...",
                                   font=self.f_status, fg=COL_MUTE, bg=COL_PANEL)
        self.lbl_status.pack(side="left", pady=6)

    # ---------- HELPER INTERFACCIA ----------
    def _set_text(self, content, muted=False):
        """Aggiorna il testo nella card della risposta."""
        self.txt.config(state="normal")
        self.txt.delete("1.0", "end")
        self.txt.insert("1.0", content)
        self.txt.tag_configure("all", foreground=COL_MUTE if muted else COL_TEXT)
        self.txt.tag_add("all", "1.0", "end")
        self.txt.config(state="disabled")

    def _status(self, text, color=COL_MUTE, dot=COL_MUTE):
        """Aggiorna la barra di stato (thread-safe via after)."""
        def upd():
            self.lbl_status.config(text=text, fg=color)
            self.dot.config(fg=dot)
        self.root.after(0, upd)

    def _ui(self, fn):
        """Esegue una funzione sul thread della GUI."""
        self.root.after(0, fn)

    # ---------- AVVIO SERVER ----------
    def _boot_server(self):
        try:
            self.llama.start()
            if self.llama.wait_ready(timeout=180):
                self._status("Modello pronto", COL_OK, COL_OK)
                self._ui(lambda: self._set_text(
                    "Premi  «PARLA»  e fai la tua domanda.", muted=True))
            else:
                self._status("Timeout avvio modello", COL_DANGER, COL_DANGER)
        except Exception as e:
            self._status("Errore avvio modello", COL_DANGER, COL_DANGER)
            self._ui(lambda: self._set_text(f"⚠  Impossibile avviare il modello:\n\n{e}"))

    # ---------- FLUSSO: PARLA ----------
    def on_talk(self):
        if self.busy:
            return
        if not self.llama.is_alive():
            self._status("Il modello non e' ancora pronto", COL_DANGER, COL_DANGER)
            return
        self.busy = True
        self._set_busy_ui(True)
        threading.Thread(target=self._talk_pipeline, daemon=True).start()

    def _set_busy_ui(self, busy):
        state = "disabled" if busy else "normal"
        self._ui(lambda: self.btn_talk.config(
            state=state,
            text="🎙   ...IN ASCOLTO" if busy else "🎙   PARLA"))
        if busy:
            self._ui(lambda: self.btn_listen.config(state="disabled"))

    def _talk_pipeline(self):
        """Registra -> trascrive -> interroga il modello. Gira in un thread."""
        try:
            with tempfile.TemporaryDirectory() as tmp:
                wav = os.path.join(tmp, "input.wav")

                # 1) REGISTRAZIONE
                self._status(f"Registrazione ({self.RECORD_SECONDS}s)... parla ora",
                             COL_ACCENT2, COL_ACCENT2)
                self._record_audio(wav)

                # 2) TRASCRIZIONE con whisper
                self._status("Trascrizione in corso...", COL_ACCENT, COL_ACCENT)
                question = self._transcribe(wav)

                if not question:
                    self._status("Non ho capito nulla, riprova", COL_DANGER, COL_DANGER)
                    self._ui(lambda: self._set_text(
                        "Non ho rilevato parole. Riprova premendo «PARLA».", muted=True))
                    return

                self._ui(lambda: self.lbl_heard.config(text=f"hai detto: \u201c{question}\u201d"))

                # 3) INTERROGAZIONE DEL MODELLO
                self._status("Il modello sta pensando...", COL_ACCENT, COL_ACCENT)
                answer = self.llama.ask(question)
                if not answer:
                    answer = "(Il modello non ha prodotto una risposta.)"

                self.last_answer = answer
                self._ui(lambda: self._set_text(answer))
                self._ui(lambda: self.btn_listen.config(state="normal"))
                self._status("Pronto", COL_OK, COL_OK)

        except Exception as e:
            self._status("Errore", COL_DANGER, COL_DANGER)
            msg = str(e)
            self._ui(lambda: self._set_text(f"⚠  Si e' verificato un errore:\n\n{msg}"))
        finally:
            self.busy = False
            self._set_busy_ui(False)

    def _record_audio(self, wav_path):
        """Registra dal microfono con arecord (16kHz mono, formato per whisper)."""
        cmd = [
            "arecord",
            "-D", ARECORD_DEVICE,
            "-f", "S16_LE",
            "-r", "16000",
            "-c", "1",
            "-d", str(self.RECORD_SECONDS),
            wav_path,
        ]
        self.rec_proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.rec_proc.wait()
        self.rec_proc = None
        if not os.path.exists(wav_path) or os.path.getsize(wav_path) < 1000:
            raise RuntimeError(
                "Registrazione fallita. Controlla il microfono e ARECORD_DEVICE "
                "(usa `arecord -l` per vedere i dispositivi).")

    def _transcribe(self, wav_path):
        """Esegue whisper-cli sul file wav e restituisce il testo trascritto."""
        if not os.path.exists(WHISPER):
            raise FileNotFoundError(f"whisper-cli non trovato:\n{WHISPER}")
        out_base = wav_path + ".out"
        cmd = [
            WHISPER,
            "-m", WHISPER_MODEL,
            "-f", wav_path,
            "-l", WHISPER_LANG,
            "-otxt",
            "-of", out_base,
            "-np",        # niente stampe di sistema
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, timeout=180)
        txt_file = out_base + ".txt"
        if os.path.exists(txt_file):
            with open(txt_file, "r", encoding="utf-8") as f:
                return f.read().strip()
        return ""

    # ---------- FLUSSO: ASCOLTA ----------
    def on_listen(self):
        if not self.last_answer:
            return
        self._ui(lambda: self.btn_listen.config(state="disabled", text="🔊   ...LETTURA"))
        threading.Thread(target=self._speak, daemon=True).start()

    def _speak(self):
        """Legge la risposta ad alta voce con espeak-ng."""
        try:
            subprocess.run(
                ["espeak-ng", "-v", TTS_VOICE, "-s", str(TTS_SPEED),
                 self.last_answer],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=120,
            )
        except FileNotFoundError:
            self._status("espeak-ng non installato (sudo apt install espeak-ng)",
                         COL_DANGER, COL_DANGER)
        except Exception as e:
            self._status(f"Errore TTS: {e}", COL_DANGER, COL_DANGER)
        finally:
            self._ui(lambda: self.btn_listen.config(state="normal", text="🔊   ASCOLTA"))

    # ---------- USCITA ----------
    def quit_app(self):
        """Chiude tutto in modo pulito: registrazione, server, finestra."""
        try:
            if self.rec_proc and self.rec_proc.poll() is None:
                self.rec_proc.terminate()
        except Exception:
            pass
        self._status("Chiusura...", COL_MUTE, COL_MUTE)
        threading.Thread(target=self._shutdown, daemon=True).start()

    def _shutdown(self):
        self.llama.stop()
        self._ui(self.root.destroy)


# ============================================================
# MAIN
# ============================================================
def main():
    root = tk.Tk()
    app = VoiceAssistantApp(root)
    # Chiusura pulita anche con segnali da terminale (Ctrl+C)
    signal.signal(signal.SIGINT,  lambda s, f: app.quit_app())
    signal.signal(signal.SIGTERM, lambda s, f: app.quit_app())
    root.protocol("WM_DELETE_WINDOW", app.quit_app)
    root.mainloop()


if __name__ == "__main__":
    main()
