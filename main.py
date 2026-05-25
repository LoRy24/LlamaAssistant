import tkinter as tk
import subprocess
import threading
import os

BASE = "/home/pi"

WHISPER = f"{BASE}/whisper.cpp/build/bin/whisper-cli"
LLAMA = f"{BASE}/llama.cpp/build/bin/llama-cli"

WHISPER_MODEL = f"{BASE}/whisper.cpp/models/ggml-base.bin"
LLAMA_MODEL = f"{BASE}/llama.cpp/models/qwen1.5b.gguf"

AUDIO_FILE = "/tmp/input.wav"


# ---------- LOGICA AI ----------
def run_cmd(cmd):
    return subprocess.check_output(cmd, shell=True).decode("utf-8", errors="ignore")


def process():
    set_status("🎤 Sto ascoltando...")

    os.system(f"arecord -f cd -d 5 {AUDIO_FILE}")

    set_status("🧠 Capisco...")

    text = run_cmd(
        f"{WHISPER} -m {WHISPER_MODEL} -f {AUDIO_FILE} -l it -nt | tail -n 1"
    ).strip()

    set_user(text)

    set_status("🤖 Penso...")

    response = run_cmd(
        f"{LLAMA} -m {LLAMA_MODEL} -p \"{text}\" -n 120 | tail -n 20"
    ).strip()

    set_ai(response)

    set_status("🔊 Parlo...")

    os.system(
        f'echo "{response}" | piper '
        f'--model ~/piper/it_IT-riccardo-x_low.onnx '
        f'--output_raw | aplay -r 22050 -f S16_LE -t raw -'
    )

    set_status("Pronto")


def start_thread():
    threading.Thread(target=process, daemon=True).start()


# ---------- UI UPDATE SAFE ----------
def set_status(t): status_var.set(t)
def set_user(t): user_var.set(f"🧑 Tu: {t}")
def set_ai(t): ai_var.set(f"🤖 AI: {t}")


# ---------- GUI ----------
root = tk.Tk()
root.title("Voice AI")

# FULLSCREEN
root.attributes("-fullscreen", True)
root.configure(bg="#0f111a")

# ESC per uscire
root.bind("<Escape>", lambda e: root.destroy())

# FONT STYLE (semplice ma pulito)
FONT_TITLE = ("Helvetica", 28, "bold")
FONT_TEXT = ("Helvetica", 18)
FONT_SMALL = ("Helvetica", 14)

# TITLE
title = tk.Label(
    root,
    text="VOICE AI ASSISTANT",
    font=FONT_TITLE,
    fg="white",
    bg="#0f111a"
)
title.pack(pady=30)

# BUTTON
btn = tk.Button(
    root,
    text="🎤 PARLA",
    font=("Helvetica", 22, "bold"),
    bg="#3a86ff",
    fg="white",
    activebackground="#265df2",
    activeforeground="white",
    height=2,
    width=18,
    command=start_thread
)
btn.pack(pady=20)

# STATUS
status_var = tk.StringVar()
status = tk.Label(
    root,
    textvariable=status_var,
    font=FONT_SMALL,
    fg="#aaaaaa",
    bg="#0f111a"
)
status.pack(pady=10)

# USER TEXT
user_var = tk.StringVar()
user = tk.Label(
    root,
    textvariable=user_var,
    font=FONT_TEXT,
    fg="#4cc9f0",
    bg="#0f111a",
    wraplength=900,
    justify="left"
)
user.pack(pady=20)

# AI TEXT
ai_var = tk.StringVar()
ai = tk.Label(
    root,
    textvariable=ai_var,
    font=FONT_TEXT,
    fg="#80ed99",
    bg="#0f111a",
    wraplength=900,
    justify="left"
)
ai.pack(pady=20)

# FOOTER
footer = tk.Label(
    root,
    text="ESC per uscire",
    font=FONT_SMALL,
    fg="#666",
    bg="#0f111a"
)
footer.pack(side="bottom", pady=20)

root.mainloop()
