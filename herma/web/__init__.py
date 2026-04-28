"""Flask web server for Herma. Routes are split across submodules."""

import functools

from flask import Flask, jsonify, request as flask_request
from flask_cors import CORS

from .. import state

flask_app = Flask(__name__)
CORS(flask_app)


def localhost_or_api_key(f):
    """Allow localhost callers; require ``X-API-Key`` header for everyone else."""
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if flask_request.remote_addr in ("127.0.0.1", "::1"):
            return f(*args, **kwargs)
        key = (flask_request.headers.get("X-API-Key")
               or flask_request.args.get("api_key"))
        if key != state.API_KEY:
            return jsonify({"error": "Invalid or missing API key"}), 401
        return f(*args, **kwargs)
    return decorated


# Importing these modules registers their blueprints / routes against ``flask_app``.
from . import lifecycle, recording, chat, sentences  # noqa: E402,F401

flask_app.register_blueprint(lifecycle.bp)
flask_app.register_blueprint(recording.bp)
flask_app.register_blueprint(chat.bp)
flask_app.register_blueprint(sentences.bp)


def run_web_server():
    from .. import config
    flask_app.run(host=config.WEB_HOST, port=config.WEB_PORT,
                  threaded=True, use_reloader=False)
