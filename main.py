#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Assistente vocale offline per Raspberry Pi
==========================================
Parla con un modello AI (Qwen 1.5B) gestito da llama.cpp.

Catena: microfono -> whisper.cpp (speech-to-text) -> llama-server (AI) -> espeak-ng (text-to-speech)

Dipendenze di sistema:
    sudo apt install espeak-ng alsa-utils python3-tk

llama-server deve essere compilato insieme a llama.cpp
(di solito in llama.cpp/build/bin/llama-server).
"""

import os
import sys
import json
import time
import signal
import threading
import subprocess
import tempfile
import shutil
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

LLAMA_HOST    = "127.0.0.1"
LLAMA_PORT    = 8080
LLAMA_THREADS = 4
LLAMA_CTX     = 2048

ARECORD_DEVICE = "default"   # "default" oppure es. "plughw:1,0" -- vedi `arecord -l`
WHISPER_LANG   = "it"

TTS_VOICE = "it"
TTS_SPEED = 160

SYSTEM_PROMPT = (
    "Sei un assistente vocale gentile e conciso. "
    "Rispondi in italiano in modo chiaro e breve, adatto alla lettura ad alta voce."
)

# ============================================================
# PALETTE  -  tema scuro caldo (carbone/talpa, accento salvia)
# ============================================================
COL_BG      = "#21201d"   # carbone caldo
COL_PANEL   = "#2b2a26"   # pannelli
COL_CARD    = "#302e2a"   # card della risposta
COL_LINE    = "#3d3a35"   # bordi sottili
COL_ACCENT  = "#a8c0a0"   # salvia tenue
COL_ACCENT2 = "#d8a48f"   # terracotta soft
COL_TEXT    = "#ece8e1"   # testo principale (avorio)
COL_MUTE    = "#938d82"   # testo secondario
COL_DIM     = "#5f5b53"   # indicatore spento
COL_OK      = "#8bb48a"   # verde soft
COL_WARN    = "#d8a48f"   # ambra/terracotta
COL_ERR     = "#c98a82"   # rosso soft (non aggressivo)


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
        try:
            with urllib.request.urlopen(self.url_health, timeout=2) as r:
                data = json.loads(r.read().decode("utf-8"))
                return data.get("status") in ("ok", "no slot available")
        except Exception:
            return False

    def start(self):
        if self.is_alive():
            return
        if not os.path.exists(LLAMA_SERVER):
            raise FileNotFoundError(
                f"llama-server non trovato in:\n{LLAMA_SERVER}")
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
        self.proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid)

    def wait_ready(self, timeout=180):
        start = time.time()
        while time.time() - start < timeout:
            if self.is_alive():
                return True
            if self.proc and self.proc.poll() is not None:
                raise RuntimeError("llama-server si e' chiuso inaspettatamente.")
            time.sleep(1)
        return False

    def ask(self, user_text):
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
            self.url_completion, data=payload,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=180) as r:
            data = json.loads(r.read().decode("utf-8"))
        return data.get("content", "").strip()

    def stop(self):
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
    RECORD_SECONDS = 8

    def __init__(self, root):
        self.root = root
        self.llama = LlamaServer()
        self.busy = False
        self.last_answer = ""
        self.rec_proc = None
        self.indicators = {}   # nome -> pallino

        self._build_ui()
        threading.Thread(target=self._boot_server, daemon=True).start()
        self._schedule_health_check()

    # ---------- COSTRUZIONE INTERFACCIA ----------
    def _build_ui(self):
        self.root.title("Assistente Vocale")
        self.root.configure(bg=COL_BG)
        self.root.attributes("-fullscreen", True)
        self.root.bind("<Escape>", lambda e: self.quit_app())

        self.f_title  = tkfont.Font(family="DejaVu Sans", size=26, weight="normal")
        self.f_sub    = tkfont.Font(family="DejaVu Sans", size=11)
        self.f_body   = tkfont.Font(family="DejaVu Serif", size=17)
        self.f_btn    = tkfont.Font(family="DejaVu Sans", size=17)
        self.f_small  = tkfont.Font(family="DejaVu Sans", size=10)
        self.f_ind    = tkfont.Font(family="DejaVu Sans", size=10)

        # --- Barra superiore (minimale) ---
        topbar = tk.Frame(self.root, bg=COL_BG)
        topbar.pack(fill="x", padx=48, pady=(34, 18))

        title_wrap = tk.Frame(topbar, bg=COL_BG)
        title_wrap.pack(side="left")
        tk.Label(title_wrap, text="oracolo", font=self.f_title,
                 fg=COL_TEXT, bg=COL_BG).pack(anchor="w")
        tk.Label(title_wrap, text="assistente vocale offline",
                 font=self.f_sub, fg=COL_MUTE, bg=COL_BG).pack(anchor="w")

        self.btn_close = tk.Button(
            topbar, text="chiudi", font=self.f_btn,
            fg=COL_MUTE, bg=COL_BG, activebackground=COL_PANEL,
            activeforeground=COL_ERR, relief="flat", bd=0,
            padx=16, pady=8, cursor="hand2",
            highlightthickness=0, command=self.quit_app,
        )
        self.btn_close.pack(side="right")

        # --- Area centrale: la risposta ---
        center = tk.Frame(self.root, bg=COL_BG)
        center.pack(fill="both", expand=True, padx=48, pady=(0, 22))

        card = tk.Frame(center, bg=COL_CARD, highlightthickness=1,
                        highlightbackground=COL_LINE)
        card.pack(fill="both", expand=True)

        cap = tk.Frame(card, bg=COL_CARD)
        cap.pack(fill="x", padx=30, pady=(24, 4))
        tk.Label(cap, text="risposta", font=self.f_small,
                 fg=COL_ACCENT, bg=COL_CARD).pack(side="left")
        self.lbl_heard = tk.Label(cap, text="", font=self.f_small,
                                  fg=COL_MUTE, bg=COL_CARD)
        self.lbl_heard.pack(side="right")

        txt_wrap = tk.Frame(card, bg=COL_CARD)
        txt_wrap.pack(fill="both", expand=True, padx=30, pady=(0, 26))

        scroll = tk.Scrollbar(txt_wrap, troughcolor=COL_CARD, bd=0,
                              highlightthickness=0)
        scroll.pack(side="right", fill="y")

        self.txt = tk.Text(
            txt_wrap, font=self.f_body, fg=COL_TEXT, bg=COL_CARD,
            wrap="word", relief="flat", bd=0, padx=2, pady=6,
            insertbackground=COL_TEXT, yscrollcommand=scroll.set,
            highlightthickness=0, spacing3=6,
        )
        self.txt.pack(side="left", fill="both", expand=True)
        scroll.config(command=self.txt.yview)
        self._set_text("Premi  parla  e fai la tua domanda.", muted=True)
        self.txt.config(state="disabled")

        # --- Pulsanti azione ---
        bottom = tk.Frame(self.root, bg=COL_BG)
        bottom.pack(fill="x", padx=48, pady=(0, 20))

        self.btn_talk = tk.Button(
            bottom, text="parla", font=self.f_btn,
            fg=COL_BG, bg=COL_ACCENT, activebackground="#bcd0b4",
            activeforeground=COL_BG, relief="flat", bd=0,
            padx=40, pady=18, cursor="hand2", highlightthickness=0,
            command=self.on_talk,
        )
        self.btn_talk.pack(side="left", expand=True, fill="x", padx=(0, 8))

        self.btn_listen = tk.Button(
            bottom, text="ascolta", font=self.f_btn,
            fg=COL_TEXT, bg=COL_PANEL, activebackground=COL_LINE,
            activeforeground=COL_TEXT, relief="flat", bd=0,
            padx=40, pady=18, cursor="hand2", highlightthickness=0,
            state="disabled", command=self.on_listen,
        )
        self.btn_listen.pack(side="left", expand=True, fill="x", padx=(8, 0))

        # --- Riga di stato testuale ---
        self.lbl_status = tk.Label(self.root, text="avvio del modello...",
                                   font=self.f_small, fg=COL_MUTE, bg=COL_BG)
        self.lbl_status.pack(pady=(0, 6))

        # --- Indicatori di stato dei servizi (in fondo) ---
        statusbar = tk.Frame(self.root, bg=COL_PANEL)
        statusbar.pack(fill="x", side="bottom")
        inner = tk.Frame(statusbar, bg=COL_PANEL)
        inner.pack(pady=12)

        for name in ("modello", "whisper", "microfono"):
            self._make_indicator(inner, name)

    def _make_indicator(self, parent, name):
        """Crea un indicatore: pallino + etichetta."""
        cell = tk.Frame(parent, bg=COL_PANEL)
        cell.pack(side="left", padx=22)
        dot = tk.Label(cell, text="\u25cf", font=self.f_ind,
                       fg=COL_DIM, bg=COL_PANEL)
        dot.pack(side="left", padx=(0, 7))
        lbl = tk.Label(cell, text=name, font=self.f_ind,
                       fg=COL_MUTE, bg=COL_PANEL)
        lbl.pack(side="left")
        self.indicators[name] = dot

    # ---------- HELPER INTERFACCIA ----------
    def _set_indicator(self, name, color):
        """Cambia il colore di un indicatore (thread-safe)."""
        dot = self.indicators.get(name)
        if dot is not None:
            self.root.after(0, lambda: dot.config(fg=color))

    def _set_text(self, content, muted=False):
        self.txt.config(state="normal")
        self.txt.delete("1.0", "end")
        self.txt.insert("1.0", content)
        self.txt.tag_configure("all", foreground=COL_MUTE if muted else COL_TEXT)
        self.txt.tag_add("all", "1.0", "end")
        self.txt.config(state="disabled")

    def _status(self, text, color=COL_MUTE):
        self.root.after(0, lambda: self.lbl_status.config(text=text, fg=color))

    def _ui(self, fn):
        self.root.after(0, fn)

    # ---------- CONTROLLO STATO SERVIZI ----------
    def _schedule_health_check(self):
        """Avvia un controllo periodico dello stato dei servizi."""
        threading.Thread(target=self._check_services, daemon=True).start()
        self.root.after(5000, self._schedule_health_check)

    def _check_services(self):
        """Verifica modello, whisper e microfono; aggiorna gli indicatori."""
        # Modello
        self._set_indicator("modello", COL_OK if self.llama.is_alive() else COL_ERR)
        # Whisper: basta che il binario e il modello esistano
        whisper_ok = os.path.exists(WHISPER) and os.path.exists(WHISPER_MODEL)
        self._set_indicator("whisper", COL_OK if whisper_ok else COL_ERR)
        # Microfono: arecord presente e almeno una scheda di cattura
        self._set_indicator("microfono", COL_OK if self._mic_available() else COL_ERR)

    def _mic_available(self):
        """True se arecord esiste e rileva un dispositivo di cattura."""
        if shutil.which("arecord") is None:
            return False
        try:
            out = subprocess.run(["arecord", "-l"], capture_output=True,
                                 text=True, timeout=5)
            return "card" in out.stdout
        except Exception:
            return False

    # ---------- AVVIO SERVER ----------
    def _boot_server(self):
        self._set_indicator("modello", COL_WARN)  # in avvio
        try:
            self.llama.start()
            if self.llama.wait_ready(timeout=180):
                self._set_indicator("modello", COL_OK)
                self._status("modello pronto", COL_OK)
                self._ui(lambda: self._set_text(
                    "Premi  parla  e fai la tua domanda.", muted=True))
            else:
                self._set_indicator("modello", COL_ERR)
                self._status("timeout avvio modello", COL_ERR)
        except Exception as e:
            self._set_indicator("modello", COL_ERR)
            self._status("errore avvio modello", COL_ERR)
            self._ui(lambda: self._set_text(
                f"Impossibile avviare il modello:\n\n{e}"))

    # ---------- FLUSSO: PARLA ----------
    def on_talk(self):
        if self.busy:
            return
        if not self.llama.is_alive():
            self._status("il modello non e' ancora pronto", COL_ERR)
            return
        self.busy = True
        self._set_busy_ui(True)
        threading.Thread(target=self._talk_pipeline, daemon=True).start()

    def _set_busy_ui(self, busy):
        state = "disabled" if busy else "normal"
        self._ui(lambda: self.btn_talk.config(
            state=state, text="in ascolto..." if busy else "parla"))
        if busy:
            self._ui(lambda: self.btn_listen.config(state="disabled"))

    def _talk_pipeline(self):
        try:
            with tempfile.TemporaryDirectory() as tmp:
                wav = os.path.join(tmp, "input.wav")

                # 1) REGISTRAZIONE
                self._status(f"registrazione ({self.RECORD_SECONDS}s)... parla ora",
                             COL_ACCENT2)
                self._record_audio(wav)

                # 2) TRASCRIZIONE
                self._status("trascrizione in corso...", COL_ACCENT)
                question = self._transcribe(wav)
                if not question:
                    self._status("non ho capito nulla, riprova", COL_ERR)
                    self._ui(lambda: self._set_text(
                        "Non ho rilevato parole. Riprova premendo  parla.",
                        muted=True))
                    return
                self._ui(lambda: self.lbl_heard.config(
                    text=f"hai detto: \u201c{question}\u201d"))

                # 3) MODELLO
                self._status("il modello sta pensando...", COL_ACCENT)
                answer = self.llama.ask(question)
                if not answer:
                    answer = "(Il modello non ha prodotto una risposta.)"
                self.last_answer = answer
                self._ui(lambda: self._set_text(answer))
                self._ui(lambda: self.btn_listen.config(state="normal"))
                self._status("pronto", COL_OK)
        except Exception as e:
            self._status("errore", COL_ERR)
            msg = str(e)
            self._ui(lambda: self._set_text(
                f"Si e' verificato un errore:\n\n{msg}"))
        finally:
            self.busy = False
            self._set_busy_ui(False)

    def _record_audio(self, wav_path):
        cmd = [
            "arecord", "-D", ARECORD_DEVICE,
            "-f", "S16_LE", "-r", "16000", "-c", "1",
            "-d", str(self.RECORD_SECONDS), wav_path,
        ]
        self.rec_proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.rec_proc.wait()
        self.rec_proc = None
        if not os.path.exists(wav_path) or os.path.getsize(wav_path) < 1000:
            raise RuntimeError(
                "Registrazione fallita. Controlla il microfono e ARECORD_DEVICE "
                "(usa `arecord -l`).")

    def _transcribe(self, wav_path):
        if not os.path.exists(WHISPER):
            raise FileNotFoundError(f"whisper-cli non trovato:\n{WHISPER}")
        out_base = wav_path + ".out"
        cmd = [
            WHISPER, "-m", WHISPER_MODEL, "-f", wav_path,
            "-l", WHISPER_LANG, "-otxt", "-of", out_base, "-np",
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
        self._ui(lambda: self.btn_listen.config(
            state="disabled", text="lettura..."))
        threading.Thread(target=self._speak, daemon=True).start()

    def _speak(self):
        try:
            subprocess.run(
                ["espeak-ng", "-v", TTS_VOICE, "-s", str(TTS_SPEED),
                 self.last_answer],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=120)
        except FileNotFoundError:
            self._status("espeak-ng non installato", COL_ERR)
        except Exception as e:
            self._status(f"errore tts: {e}", COL_ERR)
        finally:
            self._ui(lambda: self.btn_listen.config(
                state="normal", text="ascolta"))

    # ---------- USCITA ----------
    def quit_app(self):
        try:
            if self.rec_proc and self.rec_proc.poll() is None:
                self.rec_proc.terminate()
        except Exception:
            pass
        self._status("chiusura...", COL_MUTE)
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
    signal.signal(signal.SIGINT,  lambda s, f: app.quit_app())
    signal.signal(signal.SIGTERM, lambda s, f: app.quit_app())
    root.protocol("WM_DELETE_WINDOW", app.quit_app)
    root.mainloop()


if __name__ == "__main__":
    main()
