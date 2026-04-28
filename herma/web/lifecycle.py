"""Lifecycle endpoints: /api/begin, /api/next, /api/end."""

from flask import Blueprint, jsonify

from .. import state
from . import localhost_or_api_key

bp = Blueprint("lifecycle", __name__)


@bp.post("/api/end")
@localhost_or_api_key
def api_end():
    print("received /api/end")
    state.request_restart()
    print("API: End requested - restarting to initial state")
    return jsonify({"status": "ok", "action": "restart"})


@bp.post("/api/begin")
@localhost_or_api_key
def api_begin():
    with state.intro_lock:
        if state.intro_state == "logo":
            state.intro_state = "instructions"
            print("API: Transitioning from logo to instructions")
            return jsonify({"status": "ok", "action": "show_instructions",
                            "state": state.intro_state})
        return jsonify({"status": "error", "message": "Not in logo state",
                        "current_state": state.intro_state}), 400


@bp.post("/api/next")
@localhost_or_api_key
def api_next():
    with state.intro_lock:
        if state.intro_state == "instructions":
            state.intro_state = "running"
            print("API: Transitioning from instructions to running")
            return jsonify({"status": "ok", "action": "start_webcam",
                            "state": state.intro_state})
        return jsonify({"status": "error", "message": "Not in instructions state",
                        "current_state": state.intro_state}), 400
