# Deployment Notes

This app is not compatible with GitHub Pages because GitHub Pages only hosts static files and cannot run:

- Streamlit
- Python dependencies like FAISS and SentenceTransformers
- a local Ollama server

## Fastest working hosting options

- Streamlit Community Cloud
- Render
- Railway

## Required runtime configuration

The app now supports environment-based configuration:

- `DATA_DIR`
- `FAISS_INDEX_PATH`
- `METADATA_PATH`
- `OLLAMA_URL`
- `OLLAMA_MODEL`
- `AUTHORITY_ALPHA`

Examples:

```powershell
$env:DATA_DIR="D:\"
$env:OLLAMA_URL="http://localhost:11434/api/generate"
streamlit run app.py
```

## Important blocker for cloud deployment

The repository does not include:

- `arxiv_faiss_index.bin`
- `arxiv_metadata.pkl`

The current app also expects an Ollama-compatible model endpoint.

That means online deployment requires one of these:

1. Mount or upload the FAISS and metadata files to the hosting environment.
2. Point the app to those files with `DATA_DIR` or explicit file-path variables.
3. Expose a reachable remote `OLLAMA_URL`, or switch the app to another hosted LLM API.

## Recommended next move

Use this repo as the source of truth, then deploy on Render or Streamlit Community Cloud with:

- hosted data files
- a remote model endpoint

GitHub Pages can still be used later for a separate static landing page, but not for the RAG app itself.
