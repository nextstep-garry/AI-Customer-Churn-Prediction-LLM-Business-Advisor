"""WSGI entry point.

Development:  python wsgi.py                    (debug only if FLASK_DEBUG=1)
Windows:      waitress-serve --listen=0.0.0.0:5000 wsgi:app
Linux/Docker: gunicorn -w 2 -b 0.0.0.0:5000 wsgi:app
"""
from churn_api import create_app
from churn_api.config import Config

app = create_app()

if __name__ == "__main__":
    cfg = Config.from_env()
    app.run(host=cfg.HOST, port=cfg.PORT, debug=cfg.DEBUG)
