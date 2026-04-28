"""Chat endpoints: /api/chat, /api/clear_chat."""

import time

from flask import Blueprint, jsonify, request as flask_request

from .. import config, state
from . import localhost_or_api_key

bp = Blueprint("chat", __name__)


@bp.post("/api/chat")
@localhost_or_api_key
def api_chat():
    data = flask_request.get_json() or {}
    message = data.get("message", "Hello")
    sender = data.get("sender", "user")

    if sender not in ("organism", "user"):
        return jsonify({"error": "sender must be 'organism' or 'user'"}), 400

    with state.chat_lock:
        state.show_chat = True
        state.chat_messages.append({
            "sender": sender,
            "message": message,
            "timestamp": time.time(),
        })
        if len(state.chat_messages) > config.MAX_CHAT_MESSAGES:
            state.chat_messages = state.chat_messages[-config.MAX_CHAT_MESSAGES:]

    with state.organism_lock:
        state.show_organism = False

    print(f"API: Chat message from {sender}: {message}")
    return jsonify({"status": "ok", "action": "show_chat",
                    "sender": sender, "message": message,
                    "show_chat": True, "total_messages": len(state.chat_messages)})


@bp.post("/api/clear_chat")
@localhost_or_api_key
def api_clear_chat():
    with state.chat_lock:
        state.chat_messages = []
        state.show_chat = False
    print("API: Chat messages cleared")
    return jsonify({"status": "ok", "action": "clear_chat", "show_chat": False})
