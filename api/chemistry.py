import os
import sys
import tempfile
from pathlib import Path

from flask import Flask, jsonify, request

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from chem import CHEMISTRY_SYSTEM_PROMPT, analyze_chemistry_image
from google import genai


app = Flask(__name__)


def load_key_env():
    """Load a local .env file when running outside Vercel."""
    env_file = ROOT_DIR / ".env"
    if not env_file.exists():
        return

    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key.strip()] = value.strip().strip('"').strip("'")


def analyze_chemistry_question(question):
    """Analyze a text-only chemistry question."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY before using the chemistry assistant.")

    client = genai.Client(api_key=api_key)
    prompt = f"""{CHEMISTRY_SYSTEM_PROMPT}

User Question: {question or "Explain the chemistry concept I asked about."}"""
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=prompt,
    )
    return response.text


@app.get("/")
def health_check():
    return jsonify({"status": "Pinkman chemistry API is running"})


@app.post("/api/chemistry")
@app.post("/")
def chemistry():
    """Handle a chemistry question and an optional uploaded image."""
    question = request.form.get("question", "")
    image = request.files.get("image")
    temporary_path = None

    try:
        if image and image.filename:
            suffix = Path(image.filename).suffix or ".jpg"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary_file:
                image.save(temporary_file)
                temporary_path = temporary_file.name

        if temporary_path:
            answer = analyze_chemistry_image(temporary_path, question)
        else:
            answer = analyze_chemistry_question(question)

        return jsonify({"answer": answer})
    except Exception as error:
        return jsonify({"error": str(error)}), 500
    finally:
        if temporary_path:
            Path(temporary_path).unlink(missing_ok=True)


if __name__ == "__main__":
    load_key_env()
    app.run(host="127.0.0.1", port=8765, debug=True)
