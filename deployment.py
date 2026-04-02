import subprocess
import time

import requests
from pyngrok import conf, ngrok

from config import BASE_DIR, NGROK_TOKEN, PORT

APP_FILE = BASE_DIR / "app.py"


def main() -> None:
    if not APP_FILE.exists():
        raise FileNotFoundError(f"Missing app file: {APP_FILE}")
    if not NGROK_TOKEN:
        raise RuntimeError("Set NGROK_TOKEN before running deployment.py")

    conf.get_default().auth_token = NGROK_TOKEN

    proc = subprocess.Popen(
        [
            "streamlit",
            "run",
            str(APP_FILE),
            "--server.port",
            str(PORT),
            "--server.headless",
            "true",
            "--server.enableCORS",
            "false",
            "--server.enableXsrfProtection",
            "false",
        ]
    )

    public_url = None
    try:
        for _ in range(30):
            try:
                if requests.get(f"http://127.0.0.1:{PORT}", timeout=2).status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(2)

        public_url = ngrok.connect(PORT)
        print(f"NeuroScan AI is live at: {public_url}")
        proc.wait()
    finally:
        if public_url is not None:
            try:
                ngrok.disconnect(public_url.public_url)
            except Exception:
                pass


if __name__ == "__main__":
    main()
