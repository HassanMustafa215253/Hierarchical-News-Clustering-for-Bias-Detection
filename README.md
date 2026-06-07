# News Bias Search Engine

This project is now split into two parts:

1. `Main.ipynb` runs in Google Colab and acts as the search + bias backend.
2. The Flask website takes a user query and displays Colab's grouped results.

## Colab backend

Run `Main.ipynb` in order until the embedding search index is ready, then run the final cell:

```text
Website API Backend for Colab
```

That cell starts a Flask API with:

```text
POST /search
```

The response groups results into:

- the positive/pro Actor A group from `bias_summary["label_positive"]`
- `neutral`
- the negative/pro Actor B group from `bias_summary["label_negative"]`

Copy the printed Colab proxy URL or ngrok URL.

## Website

Install and run locally:

```bash
python -m pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000`, paste the Colab backend URL, and search.

Alternatively, set the backend URL before running the website:

```powershell
$env:COLAB_BACKEND_URL="https://your-colab-backend-url"
python app.py
```

## Architecture

```text
User query
  ↓
Website
  ↓
Colab /search API
  ↓
semantic search from Main.ipynb
  ↓
bias grouping from Main.ipynb
  ↓
grouped search results back to website
```
