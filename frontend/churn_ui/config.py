"""Frontend settings, read from the environment (see .env.example). Nothing is hard-coded."""
import os

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:5000").rstrip("/")
API_KEY = os.getenv("API_KEY", "")              # must match the backend's API_KEY when that is set
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "30"))
UPLOAD_TIMEOUT = float(os.getenv("UPLOAD_TIMEOUT", "300"))
