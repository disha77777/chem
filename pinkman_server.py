import json
import os
import tempfile
from email.parser import BytesParser
from email.policy import default
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from chem import analyze_chemistry_image


BASE_DIR = Path(__file__).resolve().parent
HTML_FILE = BASE_DIR / "pinkman-chat_1.html"


def load_key_env():
    for filename in ("key.env", ".env"):
        env_file = BASE_DIR / filename
        if not env_file.exists():
            continue
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ[key.strip()] = value.strip().strip('"').strip("'")


class PinkmanHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/pinkman-chat_1.html"):
            self.path = "/pinkman-chat_1.html"
        return super().do_GET()

    def do_POST(self):
        if self.path != "/api/chemistry":
            self.send_error(404, "Endpoint not found")
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        request_body = self.rfile.read(content_length)
        message = BytesParser(policy=default).parsebytes(
            b"Content-Type: "
            + self.headers.get("Content-Type", "").encode("utf-8")
            + b"\r\nMIME-Version: 1.0\r\n\r\n"
            + request_body
        )
        fields = {part.get_param("name", header="content-disposition"): part for part in message.walk() if part.get_content_disposition() == "form-data"}
        question_part = fields.get("question")
        question = question_part.get_content() if question_part else ""
        image_field = fields.get("image")
        temp_path = None

        try:
            if image_field is not None and image_field.get_filename():
                suffix = Path(image_field.get_filename() or "image.jpg").suffix or ".jpg"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                    temp_file.write(image_field.get_payload(decode=True) or b"")
                    temp_path = temp_file.name

            if temp_path:
                answer = analyze_chemistry_image(temp_path, question)
            else:
                answer = analyze_chemistry_question(question)

            self._send_json({"answer": answer})
        except Exception as error:
            self._send_json({"error": str(error)}, status=500)
        finally:
            if temp_path:
                Path(temp_path).unlink(missing_ok=True)

    def _send_json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def analyze_chemistry_question(question):
    """Analyze a text-only chemistry question through the same chemistry prompt."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY in .env before using chemistry assistant.")

    from google import genai

    client = genai.Client(api_key=api_key)
    prompt = f"""{chemistry_prompt()}

User Question: {question or 'Explain the chemistry concept I asked about.'}"""
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=prompt,
    )
    return response.text


def chemistry_prompt():
    from chem import CHEMISTRY_SYSTEM_PROMPT

    return CHEMISTRY_SYSTEM_PROMPT


if __name__ == "__main__":
    load_key_env()
    server = ThreadingHTTPServer(("127.0.0.1", 8765), PinkmanHandler)
    print("Pinkman chemistry chat: http://127.0.0.1:8765")
    server.serve_forever()