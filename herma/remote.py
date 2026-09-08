"""HTTP and SSE communication with HermaSentiChat / HermaUploadReceiver.

The flow after a recording stops:
    1. Upload frames or video to ``state.AWS_UPLOAD_URL``.
    2. Open SSE connection to port 5002 (``/api/analyse-video-agents``).
    3. When ``agent2b`` completes, set the organism overlay text and
       POST organism info to ``/api/chatready``.
"""

import json
from pathlib import Path
from urllib.parse import urlparse

import requests

from . import config, state


def fetch_recording_mode() -> str:
    """Fetch the recording ``input_type`` from HermaSentiChat settings.

    Returns ``'video'`` (default) or ``'image_sequence'``.
    """
    parsed = urlparse(state.AWS_UPLOAD_URL)
    url = f"http://{parsed.hostname}:5002/api/settings/recording-mode"
    try:
        resp = requests.get(url, timeout=5)
        if resp.ok:
            mode = resp.json().get('input_type', 'video')
            print(f"Recording mode from server: {mode}")
            return mode
    except Exception as e:
        print(f"Could not fetch recording mode (defaulting to 'video'): {e}")
    return 'video'


def upload_video(video_path: Path):
    if not video_path.exists():
        print(f"No video to upload: {video_path}")
        return False
    try:
        print(f"Uploading video {video_path.name}...")
        with open(video_path, "rb") as fh:
            response = requests.post(
                state.AWS_UPLOAD_URL,
                headers={"X-API-Key": state.AWS_UPLOAD_KEY},
                files={"files": (video_path.name, fh, "video/mp4")},
                data={
                    "sequence_name": video_path.parent.name,
                    "timestamp": video_path.parent.name,
                },
                timeout=120,
            )
        if response.ok:
            print(f"Upload successful: {response.json()}")
            state.set_organism_text(config.LOADING_TEXT)
            analyse_video(video_path)
            return True
        print(f"Upload failed: {response.status_code} - {response.text}")
        return False
    except requests.exceptions.RequestException as e:
        print(f"Upload error: {e}")
        return False


def upload_and_analyse_images(image_paths: list, seq_name: str):
    """Upload image frames to HermaUploadReceiver, then send to HermaSentiChat for analysis."""
    if not image_paths:
        print("No image frames to upload")
        return

    try:
        print(f"Uploading {len(image_paths)} frames to receiver as sequence '{seq_name}'...")
        open_files = []
        multipart = []
        for p in image_paths:
            fh = open(p, 'rb')
            open_files.append(fh)
            multipart.append(('files', (Path(p).name, fh, 'image/jpeg')))
        response = requests.post(
            state.AWS_UPLOAD_URL,
            headers={"X-API-Key": state.AWS_UPLOAD_KEY},
            files=multipart,
            data={
                "sequence_name": seq_name,
                "image_count": str(len(image_paths)),
                "timestamp": seq_name,
            },
            timeout=120,
        )
        for fh in open_files:
            fh.close()
        if response.ok:
            print(f"Image upload successful: {response.json()}")
        else:
            print(f"Image upload failed: {response.status_code} - {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Image upload error: {e}")

    state.set_organism_text(config.LOADING_TEXT)
    analyse_images(image_paths)


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


def analyse_images(image_paths: list):
    """Call the analyse endpoint with image frames, stream SSE, display results."""
    parsed = urlparse(state.AWS_UPLOAD_URL)
    url = f"http://{parsed.hostname}:5002/api/analyse-video-agents"
    print(f"Starting image sequence analysis ({len(image_paths)} frames)...")
    try:
        open_files = []
        multipart = []
        for p in image_paths:
            fh = open(p, 'rb')
            open_files.append(fh)
            multipart.append(('images', (Path(p).name, fh, 'image/jpeg')))
        response = requests.post(
            url,
            files=multipart,
            params={"agent2c": "false", "agent3": "false"},
            stream=True,
            timeout=300,
        )
        for fh in open_files:
            fh.close()
        if not response.ok:
            print(f"Analysis request failed: {response.status_code} - {response.text}")
            return
        _consume_analysis_stream(response, parsed)
    except requests.exceptions.RequestException as e:
        print(f"Analysis error: {e}")


def analyse_video(video_path: Path):
    """Call the video analysis endpoint, stream SSE, display results."""
    parsed = urlparse(state.AWS_UPLOAD_URL)
    url = f"http://{parsed.hostname}:5002/api/analyse-video-agents"
    print(f"Starting video analysis for: {video_path}")
    try:
        with open(video_path, "rb") as fh:
            response = requests.post(
                url,
                files={"video": (video_path.name, fh, "video/mp4")},
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
