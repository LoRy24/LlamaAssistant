#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RaspAI - Assistente vocale offline per Raspberry Pi
===================================================
Parla con un modello AI (Qwen e altri) gestito da llama.cpp.
Interfaccia ottimizzata per schermo VERTICALE 600 x 1024 px.

Catena: microfono -> whisper.cpp -> llama-server -> Piper (o espeak-ng)

--------------------------------------------------------------------
INSTALLAZIONE DELLA VOCE PIPER (voce umana, offline)
--------------------------------------------------------------------
Lo script setup.sh installa Piper e la voce automaticamente.
Per farlo a mano: la voce va nella cartella  piper/  del repository.
Se Piper o il file voce non ci sono, l'app usa automaticamente espeak-ng.
Altre voci italiane disponibili su HuggingFace (rhasspy/piper-voices):
  it_IT-riccardo-x_low  (maschile, leggera)
--------------------------------------------------------------------

Dipendenze di sistema:
    sudo apt install espeak-ng alsa-utils python3-tk
"""

import os
import re
import sys
import json
import time
import math
import signal
import threading
import subprocess
import tempfile
import shutil
import glob
import urllib.request
import urllib.error
import tkinter as tk
from tkinter import font as tkfont

# ============================================================
# CONFIGURAZIONE
# ============================================================
# I percorsi sono RELATIVI alla cartella del repository: lo script di
# installazione (setup.sh) mette tutto qui dentro, quindi il progetto e'
# autocontenuto e portabile. Si puo' forzare un'altra base con la
# variabile d'ambiente RASPAI_HOME.
BASE = os.environ.get(
    "RASPAI_HOME",
    os.path.dirname(os.path.abspath(__file__))
)

WHISPER       = f"{BASE}/whisper.cpp/build/bin/whisper-cli"
WHISPER_MODEL = f"{BASE}/whisper.cpp/models/ggml-base.bin"

LLAMA_SERVER  = f"{BASE}/llama.cpp/build/bin/llama-server"
MODELS_DIR    = f"{BASE}/models"                     # cartella scandita per i .gguf
DEFAULT_MODEL = f"{MODELS_DIR}/qwen1.5b.gguf"        # modello iniziale

LLAMA_HOST    = "127.0.0.1"
LLAMA_PORT    = 8080
LLAMA_THREADS = 4
LLAMA_CTX     = 2048

ARECORD_DEVICE = "default"   # "default" oppure es. "plughw:1,0" -- vedi `arecord -l`
WHISPER_LANG   = "it"


# --- Voce Piper (umana). Se mancano, fallback automatico a espeak-ng ---
PIPER_MODEL = f"{BASE}/piper/it_IT-paola-medium.onnx"

# --- Voce espeak-ng (ripiego robotico) ---
TTS_VOICE = "it"
TTS_SPEED = 160

# Dimensioni schermo (verticale)
SCREEN_W = 600
SCREEN_H = 1024

SYSTEM_PROMPT = (
    "Sei un assistente vocale gentile e conciso. "
    "Rispondi in italiano in modo chiaro e breve, adatto alla lettura ad alta voce."
)

# ============================================================
# PALETTE  -  scuro morbido con accenti NEON
# ============================================================
COL_BG       = "#15151f"   # blu-nero profondo morbido
COL_PANEL    = "#1d1d2b"   # pannelli
COL_CARD     = "#21212f"   # superfici card
COL_LINE     = "#2e2e42"   # bordi sottili
COL_TEXT     = "#f0eef7"   # testo principale
COL_MUTE     = "#8b88a3"   # testo secondario

COL_NEON     = "#00e5d4"   # ciano neon (accento principale)
COL_NEON2    = "#b06bff"   # viola neon (accento secondario)
COL_NEON_PK  = "#ff5fa2"   # rosa neon

COL_USER_BUB = "#2a2540"   # bolla utente (viola scuro)
COL_AI_BUB   = "#10303a"   # bolla AI (ciano scuro)

COL_OK       = "#5fffd0"   # verde-ciano (servizio ok)
COL_WARN     = "#ffcf6b"   # ambra
COL_ERR      = "#ff7a8a"   # rosso soft
COL_DIM      = "#44445c"   # indicatore spento


# ------------------------------------------------------------
# Utilita': nomi modello semplificati
# ------------------------------------------------------------
def pretty_model_name(path):
    """Trasforma un nome file .gguf in un'etichetta leggibile.
    es. 'qwen2.5-1.5b-instruct-q4_k_m.gguf' -> 'Qwen2.5 1.5b'
    """
    name = os.path.splitext(os.path.basename(path))[0]
    # togli suffissi di quantizzazione e tag tecnici comuni
    junk = ["instruct", "chat", "gguf", "ggml", "f16", "fp16", "bf16",
            "mini", "base", "it", "en"]
    junk += [f"q{n}" for n in range(2, 9)]
    # separa su -, _ e spazi; il PUNTO non separa se sta tra due cifre
    # (cosi' '2.5' resta unito ma 'v1.0.gguf' viene gestito a parte)
    tmp = re.sub(r"(?<!\d)\.(?!\d)", " ", name)   # punti non-decimali -> spazio
    parts = re.split(r"[-_\s]+", tmp)
    kept = []
    for p in parts:
        pl = p.lower()
        if not p:
            continue
        if pl in junk:
            continue
        if re.fullmatch(r"q\d(\.\d+)?(_[a-z0-9]+)*", pl):  # q4_k_m, q8_0 ...
            continue
        if re.fullmatch(r"k|m|s|l|xl|xs", pl):              # lettere di quant
            continue
        if re.fullmatch(r"v\d+(\.\d+)*", pl):               # tag versione vX.Y
            continue
        if "q" in pl and re.search(r"q\d", pl):             # token con quant dentro
            continue
        kept.append(p)
    if not kept:
        kept = [name]
    # tieni al massimo i primi 3 token: di solito 'famiglia + versione + size'
    label = " ".join(kept[:3])
    label = label[:1].upper() + label[1:]
    return label.strip()


# ============================================================
# GESTIONE DEL SERVER LLAMA
# ============================================================
class LlamaServer:
    """Avvia, riavvia e mantiene caldo llama-server in background."""

    def __init__(self, model_path):
        self.model_path = model_path
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
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Modello non trovato:\n{self.model_path}")
        cmd = [
            LLAMA_SERVER,
            "-m", self.model_path,
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

    def restart_with(self, model_path):
        """Ferma il server corrente e lo riavvia con un altro modello."""
        self.stop()
        # attendi il rilascio della porta
        for _ in range(10):
            if not self.is_alive():
                break
            time.sleep(0.5)
        self.model_path = model_path
        self.start()
        return self.wait_ready(timeout=180)

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
        self.proc = None


# ============================================================
# WIDGET: pulsante con angoli arrotondati (Canvas)
# ============================================================
class RoundButton(tk.Canvas):
    """Pulsante disegnato su Canvas con angoli arrotondati."""

    def __init__(self, parent, text, command, bg, fg,
                 font, width, height, radius=22, active_bg=None,
                 glow=None):
        super().__init__(parent, width=width, height=height,
                         bg=parent["bg"], highlightthickness=0, bd=0)
        self.command = command
        self.bg = bg
        self.fg = fg
        self.active_bg = active_bg or bg
        self.glow = glow
        self.radius = radius
        self.w, self.h = width, height
        self._font = font
        self._text = text
        self._enabled = True
        self._draw(self.bg)
        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.configure(cursor="hand2")

    def _round_rect(self, x1, y1, x2, y2, r, **kw):
        pts = [
            x1+r, y1, x2-r, y1, x2, y1, x2, y1+r, x2, y2-r, x2, y2,
            x2-r, y2, x1+r, y2, x1, y2, x1, y2-r, x1, y1+r, x1, y1,
        ]
        return self.create_polygon(pts, smooth=True, **kw)

    def _draw(self, fill):
        self.delete("all")
        pad = 3
        # alone neon morbido (se richiesto)
        if self.glow and self._enabled:
            for i, a in enumerate((6, 4, 2)):
                self._round_rect(pad-a, pad-a, self.w-pad+a, self.h-pad+a,
                                 self.radius+a, fill="", outline=self.glow,
                                 width=1)
        self._round_rect(pad, pad, self.w-pad, self.h-pad, self.radius,
                         fill=fill, outline="")
        col = self.fg if self._enabled else COL_MUTE
        self.create_text(self.w//2, self.h//2, text=self._text,
                         fill=col, font=self._font)

    def set_text(self, text):
        self._text = text
        self._draw(self.bg if self._enabled else COL_LINE)

    def set_enabled(self, enabled):
        self._enabled = enabled
        self._draw(self.bg if enabled else COL_LINE)

    def _on_press(self, _):
        if self._enabled:
            self._draw(self.active_bg)

    def _on_release(self, _):
        if self._enabled:
            self._draw(self.bg)
            if self.command:
                self.command()


# ============================================================
# APPLICAZIONE GUI
# ============================================================
class RaspAIApp:
    RECORD_SECONDS = 8

    def __init__(self, root):
        self.root = root
        self.busy = False
        self.last_answer = ""
        self.rec_proc = None
        self.indicators = {}
        self.bubbles = []           # widget delle bolle in conversazione
        self.anim_on = False        # animazione "in attesa" attiva
        self.anim_phase = 0

        # Modelli disponibili
        self.models = self._scan_models()
        self.current_model = DEFAULT_MODEL
        if not os.path.exists(self.current_model) and self.models:
            self.current_model = self.models[0]
        self.llama = LlamaServer(self.current_model)

        self.use_piper = self._piper_available()

        self._build_ui()
        threading.Thread(target=self._boot_server, daemon=True).start()
        self._schedule_health_check()
        self._tick_anim()

    # ---------- SCANSIONE MODELLI ----------
    def _scan_models(self):
        found = sorted(glob.glob(os.path.join(MODELS_DIR, "*.gguf")))
        return found

    def _piper_path(self):
        """Trova l'eseguibile piper: prima nel venv, poi nel PATH di sistema."""
        venv_piper = os.path.join(BASE, "venv", "bin", "piper")
        if os.path.exists(venv_piper):
            return venv_piper
        return shutil.which("piper")

    def _piper_available(self):
        return (self._piper_path() is not None
                and os.path.exists(PIPER_MODEL))

    # ---------- COSTRUZIONE INTERFACCIA ----------
    def _build_ui(self):
        self.root.title("RaspAI")
        self.root.configure(bg=COL_BG)
        self.root.geometry(f"{SCREEN_W}x{SCREEN_H}+0+0")
        self.root.attributes("-fullscreen", True)
        self.root.resizable(False, False)
        self.root.bind("<Escape>", lambda e: self.quit_app())

        self.f_title  = tkfont.Font(family="DejaVu Sans", size=21, weight="bold")
        self.f_sub    = tkfont.Font(family="DejaVu Sans", size=10)
        self.f_body   = tkfont.Font(family="DejaVu Sans", size=13)
        self.f_btn    = tkfont.Font(family="DejaVu Sans", size=16, weight="bold")
        self.f_mini   = tkfont.Font(family="DejaVu Sans", size=13)
        self.f_small  = tkfont.Font(family="DejaVu Sans", size=9)
        self.f_ind    = tkfont.Font(family="DejaVu Sans", size=9)
        self.f_bub    = tkfont.Font(family="DejaVu Sans", size=12)
        self.f_bublbl = tkfont.Font(family="DejaVu Sans", size=8, weight="bold")

        PAD = 20

        # === RIGA 1: barra superiore ===
        topbar = tk.Frame(self.root, bg=COL_BG)
        topbar.pack(fill="x", padx=PAD, pady=(18, 8))

        title_wrap = tk.Frame(topbar, bg=COL_BG)
        title_wrap.pack(side="left")
        tk.Label(title_wrap, text="RaspAI", font=self.f_title,
                 fg=COL_NEON, bg=COL_BG).pack(anchor="w")
        tk.Label(title_wrap, text="assistente vocale offline",
                 font=self.f_sub, fg=COL_MUTE, bg=COL_BG).pack(anchor="w")

        # gruppo pulsanti a destra: modello + chiudi
        btns = tk.Frame(topbar, bg=COL_BG)
        btns.pack(side="right")

        self.btn_model = tk.Button(
            btns, text="\u25a4 modello", font=self.f_mini,
            fg=COL_NEON2, bg=COL_PANEL, activebackground=COL_LINE,
            activeforeground=COL_NEON2, relief="flat", bd=0,
            padx=12, pady=7, cursor="hand2", highlightthickness=0,
            command=self._open_model_menu,
        )
        self.btn_model.pack(side="left", padx=(0, 8))

        self.btn_close = tk.Button(
            btns, text="\u2715", font=self.f_mini,
            fg=COL_MUTE, bg=COL_PANEL, activebackground=COL_LINE,
            activeforeground=COL_ERR, relief="flat", bd=0,
            padx=12, pady=7, cursor="hand2", highlightthickness=0,
            command=self.quit_app,
        )
        self.btn_close.pack(side="left")

        # === FONDO: indicatori di stato ===
        statusbar = tk.Frame(self.root, bg=COL_PANEL)
        statusbar.pack(fill="x", side="bottom")
        inner = tk.Frame(statusbar, bg=COL_PANEL)
        inner.pack(pady=9)
        voice_name = "piper" if self.use_piper else "espeak"
        for name in ("modello", "whisper", "microfono", voice_name):
            self._make_indicator(inner, name)

        # === riga di stato testuale ===
        self.lbl_status = tk.Label(self.root, text="avvio del modello...",
                                   font=self.f_small, fg=COL_MUTE, bg=COL_BG)
        self.lbl_status.pack(side="bottom", pady=(2, 6))

        # === pulsanti azione (arrotondati) ===
        bottom = tk.Frame(self.root, bg=COL_BG)
        bottom.pack(side="bottom", fill="x", padx=PAD, pady=(0, 6))
        bw = SCREEN_W - 2 * PAD

        self.btn_talk = RoundButton(
            bottom, text="\U0001f3a4   parla", command=self.on_talk,
            bg=COL_NEON, fg=COL_BG, active_bg="#4ff5e8",
            font=self.f_btn, width=bw, height=66, radius=26,
            glow=COL_NEON)
        self.btn_talk.pack(pady=(0, 8))

        self.btn_listen = RoundButton(
            bottom, text="\U0001f50a   ascolta", command=self.on_listen,
            bg=COL_PANEL, fg=COL_TEXT, active_bg=COL_LINE,
            font=self.f_btn, width=bw, height=66, radius=26)
        self.btn_listen.set_enabled(False)
        self.btn_listen.pack()

        # === centro: conversazione a bolle (scrollabile) ===
        center = tk.Frame(self.root, bg=COL_BG)
        center.pack(fill="both", expand=True, padx=PAD, pady=(2, 10))

        self.chat_canvas = tk.Canvas(center, bg=COL_BG, highlightthickness=0,
                                     bd=0)
        chat_scroll = tk.Scrollbar(center, command=self.chat_canvas.yview,
                                   troughcolor=COL_BG, bd=0,
                                   highlightthickness=0)
        self.chat_canvas.configure(yscrollcommand=chat_scroll.set)
        chat_scroll.pack(side="right", fill="y")
        self.chat_canvas.pack(side="left", fill="both", expand=True)

        # frame interno che contiene le bolle
        self.chat_frame = tk.Frame(self.chat_canvas, bg=COL_BG)
        self.chat_window = self.chat_canvas.create_window(
            (0, 0), window=self.chat_frame, anchor="nw")
        self.chat_frame.bind(
            "<Configure>",
            lambda e: self.chat_canvas.configure(
                scrollregion=self.chat_canvas.bbox("all")))
        self.chat_canvas.bind(
            "<Configure>",
            lambda e: self.chat_canvas.itemconfig(
                self.chat_window, width=e.width))

        self.chat_width = SCREEN_W - 2 * PAD - 16
        self._add_system_line("Premi  parla  e fai la tua domanda.")

    def _make_indicator(self, parent, name):
        cell = tk.Frame(parent, bg=COL_PANEL)
        cell.pack(side="left", padx=10)
        dot = tk.Label(cell, text="\u25cf", font=self.f_ind,
                       fg=COL_DIM, bg=COL_PANEL)
        dot.pack(side="left", padx=(0, 5))
        lbl = tk.Label(cell, text=name, font=self.f_ind,
                       fg=COL_MUTE, bg=COL_PANEL)
        lbl.pack(side="left")
        self.indicators[name] = dot

    # ---------- BOLLE DI CONVERSAZIONE ----------
    def _add_system_line(self, text):
        """Riga di testo discreta, centrata (non e' una bolla)."""
        lbl = tk.Label(self.chat_frame, text=text, font=self.f_small,
                       fg=COL_MUTE, bg=COL_BG, wraplength=self.chat_width,
                       justify="center")
        lbl.pack(pady=18)
        self.bubbles.append(lbl)
        self._scroll_to_end()

    def _add_bubble(self, who, text):
        """Aggiunge una bolla chat. who = 'user' oppure 'ai'.
        Restituisce il frame della bolla (per poterlo aggiornare)."""
        is_user = (who == "user")
        outer = tk.Frame(self.chat_frame, bg=COL_BG)
        outer.pack(fill="x", pady=6)

        bub_bg  = COL_USER_BUB if is_user else COL_AI_BUB
        accent  = COL_NEON2 if is_user else COL_NEON
        label   = "TU" if is_user else "RASPAI"

        # Canvas per disegnare la bolla arrotondata
        maxw = int(self.chat_width * 0.82)
        wrap = maxw - 28
        # misura l'altezza necessaria al testo
        meas = tk.Label(self.root, text=text, font=self.f_bub,
                        wraplength=wrap, justify="left")
        meas.update_idletasks()
        txt_h = meas.winfo_reqheight()
        txt_w = min(meas.winfo_reqwidth(), wrap)
        meas.destroy()

        bub_w = txt_w + 28
        bub_h = txt_h + 40
        cv = tk.Canvas(outer, width=bub_w, height=bub_h, bg=COL_BG,
                       highlightthickness=0, bd=0)
        cv.pack(side="right" if is_user else "left",
                padx=(0, 2) if is_user else (2, 0))

        r = 18
        pts = [
            r, 0, bub_w-r, 0, bub_w, 0, bub_w, r, bub_w, bub_h-r,
            bub_w, bub_h, bub_w-r, bub_h, r, bub_h, 0, bub_h,
            0, bub_h-r, 0, r, 0, 0,
        ]
        cv.create_polygon(pts, smooth=True, fill=bub_bg, outline=accent,
                          width=1)
        cv.create_text(14, 13, text=label, anchor="w", fill=accent,
                       font=self.f_bublbl)
        cv.create_text(14, 26, text=text, anchor="nw", fill=COL_TEXT,
                       font=self.f_bub, width=wrap)

        self.bubbles.append(outer)
        self._scroll_to_end()
        return cv

    def _add_thinking_bubble(self):
        """Bolla AI animata mentre il modello pensa. Restituisce il canvas."""
        outer = tk.Frame(self.chat_frame, bg=COL_BG)
        outer.pack(fill="x", pady=6)
        cv = tk.Canvas(outer, width=120, height=52, bg=COL_BG,
                       highlightthickness=0, bd=0)
        cv.pack(side="left", padx=(2, 0))
        r = 18
        w, h = 120, 52
        pts = [r,0, w-r,0, w,0, w,r, w,h-r, w,h, w-r,h, r,h, 0,h, 0,h-r, 0,r, 0,0]
        cv.create_polygon(pts, smooth=True, fill=COL_AI_BUB,
                          outline=COL_NEON, width=1)
        cv.create_text(14, 13, text="RASPAI", anchor="w", fill=COL_NEON,
                       font=self.f_bublbl)
        # tre puntini animati
        self._think_dots = []
        for i in range(3):
            d = cv.create_oval(34+i*22, 30, 46+i*22, 42,
                               fill=COL_NEON, outline="")
            self._think_dots.append(d)
        self._think_canvas = cv
        self.bubbles.append(outer)
        self._scroll_to_end()
        return cv

    def _scroll_to_end(self):
        self.chat_canvas.update_idletasks()
        self.chat_canvas.configure(scrollregion=self.chat_canvas.bbox("all"))
        self.chat_canvas.yview_moveto(1.0)

    # ---------- ANIMAZIONE NEON DI ATTESA ----------
    def _tick_anim(self):
        """Loop di animazione (gira sempre, agisce solo se anim_on)."""
        if self.anim_on:
            self.anim_phase += 1
            # pulsazione dei tre puntini "thinking"
            if getattr(self, "_think_canvas", None) is not None:
                cv = self._think_canvas
                for i, d in enumerate(self._think_dots):
                    # onda sinusoidale sfasata per ogni puntino
                    s = math.sin((self.anim_phase / 3.0) - i * 0.9)
                    bright = (s + 1) / 2  # 0..1
                    col = self._mix(COL_AI_BUB, COL_NEON, 0.25 + 0.75*bright)
                    try:
                        cv.itemconfig(d, fill=col)
                    except tk.TclError:
                        pass
            # bordo neon pulsante del pulsante parla
            glow = self._mix(COL_NEON, COL_NEON_PK,
                             (math.sin(self.anim_phase/4.0)+1)/2)
            self.btn_talk.glow = glow
            self.btn_talk._draw(self.btn_talk.bg
                                if self.btn_talk._enabled else COL_LINE)
        self.root.after(90, self._tick_anim)

    @staticmethod
    def _mix(hex1, hex2, t):
        """Interpola due colori esadecimali. t in [0,1]."""
        t = max(0.0, min(1.0, t))
        c1 = tuple(int(hex1[i:i+2], 16) for i in (1, 3, 5))
        c2 = tuple(int(hex2[i:i+2], 16) for i in (1, 3, 5))
        m = tuple(int(a + (b-a)*t) for a, b in zip(c1, c2))
        return f"#{m[0]:02x}{m[1]:02x}{m[2]:02x}"

    def _start_thinking_anim(self):
        self.anim_on = True
        self.anim_phase = 0

    def _stop_thinking_anim(self):
        self.anim_on = False
        self.btn_talk.glow = COL_NEON
        self._think_canvas = None
        # ripristina il pulsante
        self.root.after(0, lambda: self.btn_talk._draw(
            self.btn_talk.bg if self.btn_talk._enabled else COL_LINE))

    # ---------- HELPER INTERFACCIA ----------
    def _set_indicator(self, name, color):
        dot = self.indicators.get(name)
        if dot is not None:
            self.root.after(0, lambda: dot.config(fg=color))

    def _status(self, text, color=COL_MUTE):
        self.root.after(0, lambda: self.lbl_status.config(text=text, fg=color))

    def _ui(self, fn):
        self.root.after(0, fn)

    # ---------- MENU SELEZIONE MODELLO ----------
    def _open_model_menu(self):
        """Mostra un menu a comparsa con i modelli disponibili."""
        self.models = self._scan_models()
        menu = tk.Menu(self.root, tearoff=0, bg=COL_PANEL, fg=COL_TEXT,
                       activebackground=COL_NEON2, activeforeground=COL_BG,
                       bd=0, font=self.f_mini)
        if not self.models:
            menu.add_command(label="(nessun modello in models/)",
                             state="disabled")
        else:
            for path in self.models:
                label = pretty_model_name(path)
                mark = "  \u2714" if path == self.current_model else ""
                menu.add_command(
                    label=label + mark,
                    command=lambda p=path: self._select_model(p))
        # posiziona il menu sotto il pulsante
        x = self.btn_model.winfo_rootx()
        y = self.btn_model.winfo_rooty() + self.btn_model.winfo_height() + 4
        menu.tk_popup(x, y)

    def _select_model(self, path):
        if path == self.current_model:
            return
        if self.busy:
            self._status("attendi: operazione in corso", COL_WARN)
            return
        self.current_model = path
        name = pretty_model_name(path)
        self._add_system_line(f"cambio modello: {name}")
        self.busy = True
        self.btn_talk.set_enabled(False)
        threading.Thread(target=self._do_switch_model,
                         args=(path, name), daemon=True).start()

    def _do_switch_model(self, path, name):
        self._set_indicator("modello", COL_WARN)
        self._status(f"avvio modello {name}...", COL_NEON2)
        try:
            ok = self.llama.restart_with(path)
            if ok:
                self._set_indicator("modello", COL_OK)
                self._status(f"modello pronto: {name}", COL_OK)
            else:
                self._set_indicator("modello", COL_ERR)
                self._status("timeout avvio modello", COL_ERR)
        except Exception as e:
            self._set_indicator("modello", COL_ERR)
            self._status(f"errore: {e}", COL_ERR)
        finally:
            self.busy = False
            self._ui(lambda: self.btn_talk.set_enabled(True))

    # ---------- AVVIO SERVER ----------
    def _boot_server(self):
        self._set_indicator("modello", COL_WARN)
        try:
            self.llama.start()
            if self.llama.wait_ready(timeout=180):
                self._set_indicator("modello", COL_OK)
                name = pretty_model_name(self.current_model)
                self._status(f"modello pronto: {name}", COL_OK)
            else:
                self._set_indicator("modello", COL_ERR)
                self._status("timeout avvio modello", COL_ERR)
        except Exception as e:
            self._set_indicator("modello", COL_ERR)
            self._status("errore avvio modello", COL_ERR)
            self._ui(lambda: self._add_system_line(
                f"Impossibile avviare il modello: {e}"))

    # ---------- CONTROLLO STATO SERVIZI ----------
    def _schedule_health_check(self):
        threading.Thread(target=self._check_services, daemon=True).start()
        self.root.after(5000, self._schedule_health_check)

    def _check_services(self):
        self._set_indicator("modello", COL_OK if self.llama.is_alive() else COL_ERR)
        whisper_ok = os.path.exists(WHISPER) and os.path.exists(WHISPER_MODEL)
        self._set_indicator("whisper", COL_OK if whisper_ok else COL_ERR)
        self._set_indicator("microfono", COL_OK if self._mic_available() else COL_ERR)
        # voce
        voice_name = "piper" if self.use_piper else "espeak"
        if self.use_piper:
            self._set_indicator("piper", COL_OK)
        else:
            ok = shutil.which("espeak-ng") is not None
            self._set_indicator("espeak", COL_OK if ok else COL_ERR)

    def _mic_available(self):
        if shutil.which("arecord") is None:
            return False
        try:
            out = subprocess.run(["arecord", "-l"], capture_output=True,
                                 text=True, timeout=5)
            return "card" in out.stdout
        except Exception:
            return False

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
        self._ui(lambda: self.btn_talk.set_enabled(not busy))
        self._ui(lambda: self.btn_talk.set_text(
            "\U0001f3a4   in ascolto..." if busy else "\U0001f3a4   parla"))
        if busy:
            self._ui(lambda: self.btn_listen.set_enabled(False))

    def _talk_pipeline(self):
        thinking_cv = None
        try:
            with tempfile.TemporaryDirectory() as tmp:
                wav = os.path.join(tmp, "input.wav")

                # 1) REGISTRAZIONE
                self._status(f"registrazione ({self.RECORD_SECONDS}s)... parla ora",
                             COL_NEON_PK)
                self._record_audio(wav)

                # 2) TRASCRIZIONE
                self._status("trascrizione in corso...", COL_NEON)
                question = self._transcribe(wav)
                if not question:
                    self._status("non ho capito nulla, riprova", COL_ERR)
                    self._ui(lambda: self._add_system_line(
                        "Non ho rilevato parole. Riprova premendo  parla."))
                    return

                # bolla utente
                self._ui(lambda: self._add_bubble("user", question))

                # 3) MODELLO  -- bolla animata di attesa
                self._status("RaspAI sta pensando...", COL_NEON)
                holder = {}
                def make_think():
                    holder["cv"] = self._add_thinking_bubble()
                self._ui(make_think)
                time.sleep(0.05)  # lascia creare il widget
                self._start_thinking_anim()

                answer = self.llama.ask(question)
                if not answer:
                    answer = "(Nessuna risposta dal modello.)"
                self.last_answer = answer

                # rimuovi la bolla "thinking" e metti la risposta
                self._stop_thinking_anim()
                def swap():
                    cv = holder.get("cv")
                    if cv is not None:
                        parent = cv.master
                        parent.destroy()
                        if parent in self.bubbles:
                            self.bubbles.remove(parent)
                    self._add_bubble("ai", answer)
                self._ui(swap)

                self._ui(lambda: self.btn_listen.set_enabled(True))
                self._status("pronto", COL_OK)
        except Exception as e:
            self._stop_thinking_anim()
            self._status("errore", COL_ERR)
            msg = str(e)
            self._ui(lambda: self._add_system_line(f"Errore: {msg}"))
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
                "Registrazione fallita. Controlla il microfono e ARECORD_DEVICE.")

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
        self._ui(lambda: self.btn_listen.set_enabled(False))
        self._ui(lambda: self.btn_listen.set_text("\U0001f50a   lettura..."))
        threading.Thread(target=self._speak, daemon=True).start()

    def _speak(self):
        """Legge la risposta. Usa Piper se disponibile, altrimenti espeak-ng."""
        try:
            if self.use_piper:
                self._speak_piper(self.last_answer)
            else:
                self._speak_espeak(self.last_answer)
        except FileNotFoundError:
            self._status("motore voce non trovato", COL_ERR)
        except Exception as e:
            self._status(f"errore voce: {e}", COL_ERR)
        finally:
            self._ui(lambda: self.btn_listen.set_enabled(True))
            self._ui(lambda: self.btn_listen.set_text("\U0001f50a   ascolta"))

    def _speak_piper(self, text):
        """Piper genera un wav, poi lo riproduce con aplay."""
        piper_bin = self._piper_path() or "piper"
        with tempfile.TemporaryDirectory() as tmp:
            wav = os.path.join(tmp, "out.wav")
            piper = subprocess.Popen(
                [piper_bin, "--model", PIPER_MODEL, "--output_file", wav],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL)
            piper.communicate(input=text.encode("utf-8"), timeout=120)
            if os.path.exists(wav):
                subprocess.run(["aplay", "-q", wav],
                               stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=120)

    def _speak_espeak(self, text):
        subprocess.run(
            ["espeak-ng", "-v", TTS_VOICE, "-s", str(TTS_SPEED), text],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=120)

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
    app = RaspAIApp(root)
    signal.signal(signal.SIGINT,  lambda s, f: app.quit_app())
    signal.signal(signal.SIGTERM, lambda s, f: app.quit_app())
    root.protocol("WM_DELETE_WINDOW", app.quit_app)
    root.mainloop()


if __name__ == "__main__":
    main()
