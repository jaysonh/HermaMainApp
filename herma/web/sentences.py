"""Sentences playback endpoints: /api/sentences, /api/sentences/stop."""

import threading
import time
from urllib.parse import urlparse

import requests
from flask import Blueprint, jsonify, request as flask_request

from .. import config, overlays, state
from . import localhost_or_api_key

bp = Blueprint("sentences", __name__)


def _page_timings(sentences, seconds_per_char):
    """Return ``(page_text, chars_per_second, typing_seconds)``."""
    page_text = "\n".join(s.strip() for s in sentences if s and s.strip())
    cps = 1.0 / seconds_per_char if seconds_per_char and seconds_per_char > 0 else 20.0
    typing_seconds = overlays.sentences_page_total_chars(page_text) / cps
    return page_text, cps, typing_seconds


def _sentences_worker(sentences, seconds_per_char):
    """Show every sentence on one page, typed out a character at a time.

    Assumes ``state.show_organism`` was already set to True by /api/stop. The
    render loop does the typing — it derives the visible character count from
    the start time this publishes — so all this thread does is wait out the
    reveal plus a hold. On finish (or stop), shows a thank-you message, sends
    /api/restart to the SentiChat backend, and triggers a local restart.
    """
    try:
        page_text, cps, typing_seconds = _page_timings(sentences, seconds_per_char)
        if not page_text:
            return

        state.start_sentences_page(page_text, cps)
        total = typing_seconds + config.SENTENCES_PAGE_HOLD
        print(f"API: Typing {len(sentences)} sentences over {typing_seconds:.1f}s "
              f"(+{config.SENTENCES_PAGE_HOLD:.0f}s hold)")

        elapsed = 0.0
        while elapsed < total:
            with state.sentences_lock:
                if state.sentences_stop:
                    break
            time.sleep(0.1)
            elapsed += 0.1
    finally:
        # Take the page down but leave the background video up — it plays under
        # the thank-you too, until the restart below resets everything.
        state.clear_sentences_page()

        with state.sentences_lock:
            was_stopped = state.sentences_stop
            state.sentences_active = False
            state.sentences_stop = False

        if not was_stopped:
            thanks = config.THANK_YOU_TEXT
            cps = _page_timings([thanks], seconds_per_char)[1]
            state.start_sentences_page(thanks, cps, align="center")
            typing = overlays.sentences_page_total_chars(thanks) / cps
            print(f"API: Typing thank-you message ({typing:.1f}s "
                  f"+{config.THANK_YOU_HOLD:.0f}s hold)")
            time.sleep(typing + config.THANK_YOU_HOLD)
            state.clear_sentences_page()

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

    _, _, typing_seconds = _page_timings(sentences, seconds_per_char)
    total_time = typing_seconds + config.SENTENCES_PAGE_HOLD
    print(f"API: Starting sentences page ({len(sentences)} sentences, ~{total_time:.1f}s total)")
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
