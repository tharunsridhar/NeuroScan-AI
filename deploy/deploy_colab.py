from __future__ import annotations

import os
import subprocess
import time

import requests
from google.colab import drive
from pyngrok import ngrok


def install_dependencies() -> None:
    subprocess.run(
        [
            "pip",
            "install",
            "-q",
            "groq",
            "fpdf2",
            "pillow",
            "streamlit",
            "pyngrok",
            "opencv-python-headless",
            "pandas",
            "requests",
        ],
        check=True,
    )
    print("Dependencies installed")


def start_streamlit(app_path: str) -> subprocess.Popen:
    subprocess.run(["pkill", "-f", "streamlit"], check=False)
    process = subprocess.Popen(
        [
            "streamlit",
            "run",
            app_path,
            "--server.port",
            "8501",
            "--server.address",
            "0.0.0.0",
        ]
    )
    time.sleep(6)
    response = requests.get("http://localhost:8501", timeout=10)
    response.raise_for_status()
    print("Streamlit is running on port 8501")
    return process


def main() -> None:
    drive.mount("/content/drive")
    groq_api_key = input("Enter GROQ API KEY: ").strip()
    ngrok_token = input("Enter NGROK TOKEN: ").strip()
    os.environ["GROQ_API_KEY"] = groq_api_key
    install_dependencies()

    app_path = "/content/drive/MyDrive/Project work/Code/app/app.py"
    if not os.path.exists(app_path):
        raise FileNotFoundError(f"App file not found at: {app_path}")

    start_streamlit(app_path)
    ngrok.kill()
    ngrok.set_auth_token(ngrok_token)
    public_url = ngrok.connect(8501)
    print("Public URL:")
    print(public_url)


if __name__ == "__main__":
    main()
