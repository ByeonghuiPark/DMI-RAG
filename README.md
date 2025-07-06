# DMI-RAG

This repository contains a simple Retrieval-Augmented Generation (RAG) example in Python. Documents placed in the `docs` folder are indexed with FAISS and used to answer questions via a locally running LlamaCpp model. A Gradio interface allows interaction through the browser.

## Requirements

Install dependencies with pip:

```bash
pip install -r requirements.txt
```

Place your GGUF model file under `./models` or set the environment variable `LLM_MODEL_PATH` to the model location. The default path is `./models/EEVE-Korean-10.8B.gguf`.

## Running

1. Put source documents into the `docs` directory.
2. Run the app:

```bash
python rag_app.py
```

The first run builds a FAISS index in `faiss_index/`. After launching, open the displayed URL to ask questions.
