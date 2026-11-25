from flask import Blueprint

v1 = Blueprint("v1", __name__, url_prefix="/v1")
ops = Blueprint("ops", __name__)  # unversioned, unauthenticated: /health, /ready

from . import chat, datasets, history, meta, predict  # noqa: E402,F401  (register routes)
