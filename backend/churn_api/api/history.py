from flask import jsonify, request

from . import v1
from .deps import ctx


@v1.get("/history")
def history():
    limit = max(1, min(request.args.get("limit", default=20, type=int), 200))
    return jsonify({"predictions": ctx().history.recent(limit)})


@v1.delete("/history")
def clear_history():
    return jsonify({"deleted": ctx().history.clear()})
