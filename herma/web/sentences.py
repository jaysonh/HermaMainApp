"""Sentences playback endpoints: /api/sentences, /api/sentences/stop."""

import threading
import time
from urllib.parse import urlparse

import requests
from flask import Blueprint, jsonify, request as flask_request

from .. import state
from . import localhost_or_api_key

bp = Blueprint("sentences", __name__)


def _sentences_worker(sentences, seconds_per_char):
    """Cycle through sentences, displaying each on the organism overlay.

    Assumes ``state.show_organism`` was already set to True by /api/stop.
    On finish (or stop), shows a thank-you message, sends /api/restart to the
    SentiChat backend, and triggers a local restart.
    """
    try:
        for sentence in sentences:
            with state.sentences_lock:
                if state.sentences_stop:
                    break

            state.set_organism_text(sentence)

            display_time = max(1.0, len(sentence) * seconds_per_char)
            print(f"API: Displaying sentence ({display_time:.1f}s): {sentence[:60]}...")

            elapsed = 0.0
            while elapsed < display_time:
                with state.sentences_lock:
                    if state.sentences_stop:
                        break
                time.sleep(0.1)
                elapsed += 0.1
    finally:
        with state.sentences_lock:
            was_stopped = state.sentences_stop
            state.sentences_active = False
            state.sentences_stop = False

        if not was_stopped:
            state.set_organism_text("Thank you for your experience")
            print("API: Showing thank-you message for 10s")
            time.sleep(10)

        try:
            parsed = urlparse(state.AWS_UPLOAD_URL)
            remote_url = f"http://{parsed.hostname}:5002/api/restart"
            resp = requests.post(remote_url,
                                 headers={"X-API-Key": state.API_KEY},
                                 timeout=5)
            print(f"API: Sent /api/restart to {remote_url} — {resp.status_code}")
        except Exception as e:
            print(f"API: Failed to send /api/restart to remote: {e}")

        state.request_restart()
        print("API: Sentences finished - restarting to initial state")


@bp.post("/api/sentences")
@localhost_or_api_key
def api_sentences():
    data = flask_request.get_json() or {}
    sentences = data.get("sentences", [])
    seconds_per_char = data.get("seconds_per_char", 0.05)

    if not isinstance(sentences, list) or len(sentences) == 0:
        return jsonify({"error": "Must provide a non-empty 'sentences' array"}), 400

    # Stop any currently running sequence and wait briefly for it to settle.
    with state.sentences_lock:
        if state.sentences_active:
            state.sentences_stop = True
    for _ in range(20):
        with state.sentences_lock:
            if not state.sentences_active:
                break
        time.sleep(0.1)

    with state.sentences_lock:
        state.sentences_active = True
        state.sentences_stop = False

    threading.Thread(
        target=_sentences_worker,
        args=(sentences, seconds_per_char),
        daemon=True,
    ).start()

    total_time = sum(max(1.0, len(s) * seconds_per_char) for s in sentences)
    print(f"API: Starting sentences display ({len(sentences)} sentences, ~{total_time:.1f}s total)")
    return jsonify({
        "status": "ok",
        "action": "sentences",
        "sentence_count": len(sentences),
        "estimated_total_seconds": round(total_time, 1),
        "seconds_per_char": seconds_per_char,
    })


@bp.post("/api/sentences/stop")
@localhost_or_api_key
def api_sentences_stop():
    with state.sentences_lock:
        if not state.sentences_active:
            return jsonify({"status": "ok", "message": "No sentences playing"})
        state.sentences_stop = True
    return jsonify({"status": "ok", "action": "sentences_stop"})
