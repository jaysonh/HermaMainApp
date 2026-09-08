"""HTTP and SSE communication with HermaSentiChat / HermaUploadReceiver.

The flow after the capture is taken:
    1. Upload the still frame to ``state.AWS_UPLOAD_URL``.
    2. Open SSE connection to port 5002 (``/api/analyse-video-agents``).
    3. When ``agent2b`` completes, set the organism overlay text and
       POST organism info to ``/api/chatready``.
"""

import json
from pathlib import Path
from urllib.parse import urlparse

import requests

from . import config, state


def upload_and_analyse_capture(image_path):
    """Upload the single captured frame to HermaUploadReceiver, then send it
    to HermaSentiChat for organ analysis."""
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"No capture to upload: {image_path}")
        return

    seq_name = image_path.parent.name
    try:
        print(f"Uploading capture {image_path.name} as '{seq_name}'...")
        with open(image_path, "rb") as fh:
            response = requests.post(
                state.AWS_UPLOAD_URL,
                headers={"X-API-Key": state.AWS_UPLOAD_KEY},
                files={"files": (image_path.name, fh, "image/jpeg")},
                data={
                    "sequence_name": seq_name,
                    "image_count": "1",
                    "timestamp": seq_name,
                },
                timeout=120,
            )
        if response.ok:
            print(f"Capture upload successful: {response.json()}")
        else:
            print(f"Capture upload failed: {response.status_code} - {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Capture upload error: {e}")

    state.set_organism_text(config.LOADING_TEXT)
    analyse_capture(image_path)


def _consume_analysis_stream(response, parsed_upload_url):
    """Read the SSE stream. When ``agent2b`` reports complete, set the
    organism overlay text and POST organism info to ``/api/chatready``."""
    for line in response.iter_lines():
        if not line:
            continue
        decoded = line.decode('utf-8')
        if not decoded.startswith('data: '):
            continue
        data = json.loads(decoded[6:])
        print(f"Analysis: {data}")

        if (data.get('stage') == 'agent2b'
                and data.get('status') == 'complete'
                and data.get('data')):
            payload = data['data']
            name = payload.get('organism_name', '')
            desc = payload.get('visual_description', '')
            sys_desc = payload.get('system_description', '')
            organism_info = {
                'organism_name': name,
                'visual_description': desc,
                'system_description': sys_desc,
            }
            if desc:
                overlay_text = f"{name}\n\n{desc}" if name else desc
                state.set_organism_text(overlay_text)
                print(f"Organism visual description set: {name}")
                try:
                    chatready_url = f"http://{parsed_upload_url.hostname}:5002/api/chatready"
                    requests.post(chatready_url, json=organism_info, timeout=5)
                    print(f"Sent /api/chatready to {chatready_url}")
                except Exception as e:
                    print(f"Failed to send /api/chatready: {e}")

        if data.get('done') or data.get('error'):
            break


def analyse_capture(image_path):
    """Call the analyse endpoint with the single captured frame, stream SSE,
    display results."""
    image_path = Path(image_path)
    parsed = urlparse(state.AWS_UPLOAD_URL)
    url = f"http://{parsed.hostname}:5002/api/analyse-video-agents"
    print(f"Starting image analysis for: {image_path}")
    # The analysis backend flattens every upload into one folder, so the run's
    # timestamp goes in the filename rather than "capture.jpg" every time.
    upload_name = f"{image_path.parent.name}{image_path.suffix}"
    try:
        with open(image_path, "rb") as fh:
            response = requests.post(
                url,
                files={"image": (upload_name, fh, "image/jpeg")},
                params={"agent2c": "false", "agent3": "false"},
                stream=True,
                timeout=300,
            )
        if not response.ok:
            print(f"Analysis request failed: {response.status_code} - {response.text}")
            return
        _consume_analysis_stream(response, parsed)
    except requests.exceptions.RequestException as e:
        print(f"Analysis error: {e}")
