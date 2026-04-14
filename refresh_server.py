#!/usr/bin/env python3
"""
refresh_server.py — Local server that triggers the screener pipeline and deploys to GitHub Pages.

Usage:
    python refresh_server.py

Then click "↻ Refresh Data" in the dashboard. The server must keep running in a terminal.
Leave it running in the background — it does nothing until the button is clicked.

Listens on http://localhost:7720
"""

import os
import sys
import subprocess
import threading
import platform
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

PORT = 7720
HERE = os.path.dirname(os.path.abspath(__file__))

# Pipeline steps: (label, command)
def _build_steps():
    python = sys.executable
    # On Windows, deploy script needs to run via Git Bash or WSL
    if platform.system() == "Windows":
        # Try git bash first, fall back to bash in PATH
        bash_candidates = [
            r"C:\Program Files\Git\bin\bash.exe",
            r"C:\Program Files (x86)\Git\bin\bash.exe",
            "bash",
        ]
        bash = next(
            (b for b in bash_candidates if os.path.isfile(b) or b == "bash"), "bash"
        )
        deploy_cmd = [bash, "deploy_dashboard.sh"]
    else:
        deploy_cmd = ["bash", "deploy_dashboard.sh"]

    return [
        ("screener",  [python, "run_screener.py"]),
        ("dashboard", [python, "generate_dashboard.py"]),
        ("deploy",    deploy_cmd),
    ]


_lock = threading.Lock()


class RefreshHandler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self._cors_headers()
        self.end_headers()

    def do_GET(self):
        if self.path == "/ping":
            self.send_response(200)
            self._cors_headers()
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"ok")

        elif self.path == "/refresh-sse":
            if not _lock.acquire(blocking=False):
                self.send_response(409)
                self._cors_headers()
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"error":"refresh already running"}')
                return

            self.send_response(200)
            self._cors_headers()
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()

            try:
                self._run_pipeline()
            finally:
                _lock.release()

        else:
            self.send_response(404)
            self._cors_headers()
            self.end_headers()

    def _run_pipeline(self):
        steps = _build_steps()
        for label, cmd in steps:
            self._sse(f"step:{label}")
            try:
                proc = subprocess.Popen(
                    cmd,
                    cwd=HERE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                # Stream output lines (useful for long screener run)
                for line in proc.stdout:
                    line = line.rstrip()
                    if line:
                        # Escape newlines so SSE stays valid
                        safe = line.replace("\n", " ").replace("\r", "")
                        self._sse(f"log:{label}:{safe}")

                proc.wait()
                if proc.returncode != 0:
                    self._sse(f"error:{label}:exited with code {proc.returncode}")
                    return
            except FileNotFoundError as exc:
                self._sse(f"error:{label}:command not found — {exc}")
                return
            except Exception as exc:
                self._sse(f"error:{label}:{exc}")
                return

            self._sse(f"done:{label}")

        self._sse("complete")

    def _sse(self, data: str):
        try:
            msg = f"data: {data}\n\n"
            self.wfile.write(msg.encode("utf-8"))
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _cors_headers(self):
        # Allow only file:// pages (Origin: null) and localhost origins.
        # Wildcard would let any webpage a user visits trigger the pipeline.
        origin = self.headers.get("Origin", "")
        allowed = origin == "null" or origin.startswith("http://localhost:") or origin.startswith("http://127.0.0.1:")
        self.send_header("Access-Control-Allow-Origin", origin if allowed else "null")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Vary", "Origin")

    def log_message(self, fmt, *args):  # silence default access log
        pass


def main():
    server = HTTPServer(("localhost", PORT), RefreshHandler)
    print(f"Refresh server listening on http://localhost:{PORT}")
    print("Open dashboard.html and click '↻ Refresh Data' to trigger the pipeline.")
    print("Press Ctrl+C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
