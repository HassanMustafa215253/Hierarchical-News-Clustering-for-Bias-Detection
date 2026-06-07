import os
from urllib.parse import urljoin

import requests
from flask import Flask, jsonify, render_template, request, Response


app = Flask(__name__)


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/favicon.ico")
def favicon():
    return Response(status=204)


def _normalise_backend_url(raw_url: str) -> str:
    backend_url = (raw_url or "").strip()
    if backend_url.endswith("/search"):
        backend_url = backend_url[:-len("/search")]
    return backend_url.rstrip("/")


@app.post("/api/search")
def search():
    payload = request.get_json(silent=True) or {}
    backend_url = _normalise_backend_url(
        payload.get("backend_url") or os.environ.get("COLAB_BACKEND_URL", "")
    )
    if not backend_url:
        return jsonify({
            "error": "Paste the Colab backend URL in the page or set COLAB_BACKEND_URL."
        }), 400

    forwarded_payload = {key: value for key, value in payload.items() if key != "backend_url"}
    try:
        response = requests.post(
            urljoin(backend_url.rstrip("/") + "/", "search"),
            json=forwarded_payload,
            timeout=300,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        return jsonify({"error": f"Could not reach Colab backend: {exc}"}), 502

    return jsonify(response.json())


if __name__ == "__main__":
    app.run(debug=True)
