# RaspAI

Assistente vocale **offline** per Raspberry Pi. Parli, un modello AI locale
(Qwen) risponde, e puoi ascoltare la risposta con una voce naturale.

Tutto gira in locale, senza connessione: la voce viene trascritta da
`whisper.cpp`, la risposta generata da un modello via `llama.cpp`, e letta
ad alta voce da Piper.

Interfaccia grafica pensata per uno schermo verticale **600 x 1024**.

## Catena di funzionamento

```
microfono -> whisper.cpp -> llama-server (Qwen) -> Piper -> altoparlante
```

## Installazione

Lo script di setup installa e compila tutto **dentro la cartella del
repository** (llama.cpp, whisper.cpp, ambiente Python, modelli, voce):

```bash
git clone <questo-repo> raspai
cd raspai
chmod +x setup.sh run.sh
./setup.sh
```

Il setup richiede `sudo` per i pacchetti di sistema e scarica circa 1 GB
fra modello AI e voce. La compilazione su Raspberry puo' durare diversi
minuti.

## Avvio

```bash
./run.sh
```

## Struttura dopo l'installazione

```
raspai/
├── main.py                # applicazione principale
├── setup.sh               # installazione
├── run.sh                 # avvio
├── requirements.txt       # pacchetti Python
├── venv/                  # ambiente virtuale Python
├── llama.cpp/             # motore AI (compilato)
├── whisper.cpp/           # motore speech-to-text (compilato)
├── piper/                 # voce text-to-speech italiana
└── models/                # modelli AI in formato .gguf
```

## Note

- Aggiungi altri modelli `.gguf` nella cartella `models/`: appariranno
  automaticamente nel menu "modello" dell'app.
- Se Piper non e' disponibile, l'app ripiega su `espeak-ng` (voce robotica).
- Se il microfono non viene rilevato, controlla `arecord -l` e modifica
  `ARECORD_DEVICE` in cima a `main.py`.
- I percorsi sono relativi al repository; per usarne altri si puo'
  impostare la variabile d'ambiente `RASPAI_HOME`.
