from pyngrok import ngrok
import http.server
import socket
import threading
import os
from urllib.parse import quote

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)
HTML_FILE = "clusters_3d.html"
ENCODED_HTML_FILE = quote(HTML_FILE)


class RootRedirectHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path in ("", "/"):
            self.send_response(302)
            self.send_header("Location", f"/{ENCODED_HTML_FILE}")
            self.end_headers()
            return
        super().do_GET()

    def list_directory(self, path):
        self.send_error(403, "Directory listing is disabled.")
        return None


def pick_server_port(preferred_port: int = 8080) -> int:
    candidates = [preferred_port] + list(range(8000, 8011))
    seen = set()
    for port in candidates:
        if port in seen:
            continue
        seen.add(port)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("No bindable local port found in range 8000-8010.")


def start_server(port: int) -> None:
    with http.server.HTTPServer(("", port), RootRedirectHandler) as httpd:
        print(f"Local server: http://127.0.0.1:{port}")
        httpd.serve_forever()


PORT = pick_server_port(8080)
thread = threading.Thread(target=start_server, args=(PORT,), daemon=False)
thread.start()

# Set your ngrok auth token (free at ngrok.com)
ngrok.set_auth_token("3D8HWdiALNTOO3eRqh7ozKx2IpR_7WijS8DUfD1Dx3ZSa8xad")

public_url = ngrok.connect(PORT)  # type: ignore
print(f"Open local: http://127.0.0.1:{PORT}")
print(f"Open public: {public_url}")

thread.join()