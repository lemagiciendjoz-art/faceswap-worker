#!/usr/bin/env python3
"""
RunPod Serverless handler - Face Swap (1 scene = 1 job).

Le worker :
  1. TELECHARGE source + refs depuis Drive (service account, lecture seule).
  2. Fait le face swap FaceFusion.
  3. ENVOIE le MP4 resultat (multipart) a un webhook n8n (result_webhook_url),
     qui l'uploade sur Drive avec l'OAuth de l'utilisateur.

Pourquoi pas d'upload Drive direct ni de base64 :
  - un service account Google n'a pas de quota pour uploader sur un My Drive perso.
  - le base64 dans la sortie RunPod depasse la limite de taille (sortie supprimee).
Donc on envoie le fichier directement a n8n (public, OAuth avec quota).
"""
import os
import json
import time
import glob
import uuid
import shutil
import tempfile
import subprocess
import urllib.request

import runpod
from google.oauth2 import service_account
from googleapiclient.discovery import build as gbuild
from googleapiclient.http import MediaIoBaseDownload

FACEFUSION_WORKDIR = os.environ.get("FACEFUSION_WORKDIR", "/app")
SWAPPER_MODEL_DEFAULT = os.environ.get("FACESWAP_SWAPPER_MODEL", "inswapper_128")
SA_SCOPE = ["https://www.googleapis.com/auth/drive"]
PER_SCENE_TIMEOUT_SECONDS = int(os.environ.get("PER_SCENE_TIMEOUT_SECONDS", "1800"))


def _drive():
    raw = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not raw:
        raise RuntimeError("GOOGLE_SERVICE_ACCOUNT_JSON env var missing")
    info = json.loads(raw)
    creds = service_account.Credentials.from_service_account_info(info, scopes=SA_SCOPE)
    return gbuild("drive", "v3", credentials=creds, cache_discovery=False)


def drive_find(svc, parent_id, name):
    q = ("name = '" + name.replace("'", "\\'") + "' and '"
         + parent_id + "' in parents and trashed = false")
    r = svc.files().list(
        q=q, fields="files(id,name)", spaces="drive",
        supportsAllDrives=True, includeItemsFromAllDrives=True,
    ).execute()
    return r.get("files", [])


def drive_download(svc, file_id, dest_path):
    req = svc.files().get_media(fileId=file_id, supportsAllDrives=True)
    with open(dest_path, "wb") as fh:
        dl = MediaIoBaseDownload(fh, req, chunksize=8 * 1024 * 1024)
        done = False
        while not done:
            _status, done = dl.next_chunk()
    return os.path.getsize(dest_path)


def post_multipart(url, fields, file_field, filename, file_bytes,
                   content_type="video/mp4", timeout=600):
    """POST multipart/form-data avec urllib (stdlib, pas de dependance)."""
    boundary = "----fsworker" + uuid.uuid4().hex
    parts = []
    for k, v in fields.items():
        parts.append(("--%s\r\nContent-Disposition: form-data; name=\"%s\"\r\n\r\n%s\r\n"
                      % (boundary, k, v)).encode("utf-8"))
    head = ("--%s\r\nContent-Disposition: form-data; name=\"%s\"; filename=\"%s\"\r\n"
            "Content-Type: %s\r\n\r\n" % (boundary, file_field, filename, content_type)).encode("utf-8")
    body = b"".join(parts) + head + file_bytes + ("\r\n--%s--\r\n" % boundary).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "multipart/form-data; boundary=%s" % boundary)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, resp.read()[:500].decode("utf-8", "replace")


def build_facefusion_command(refs_glob, target, output, options):
    options = options or {}
    swapper_model = options.get("swapper_model", SWAPPER_MODEL_DEFAULT)
    parts = [
        "cd " + FACEFUSION_WORKDIR + " && python facefusion.py headless-run",
        "--source-paths " + refs_glob,
        "--target-path " + target,
        "--output-path " + output,
        "--processors face_swapper face_enhancer",
        "--face-swapper-model " + str(swapper_model),
        "--face-enhancer-model " + str(options.get("enhancer_model", "gfpgan_1.4")),
        "--face-selector-mode " + str(options.get("selector_mode", "reference")),
        "--reference-face-distance " + str(options.get("reference_face_distance", 0.4)),
        "--face-mask-types " + " ".join(options.get("face_mask_types", ["occlusion", "box", "region"])),
        "--face-mask-blur " + str(options.get("face_mask_blur", 0.3)),
        "--output-video-encoder " + str(options.get("output_video_encoder", "libx264")),
        "--output-video-quality " + str(options.get("output_video_quality", 90)),
        "--output-video-preset " + str(options.get("output_video_preset", "slow")),
    ]
    if options.get("face_selector_gender"):
        parts.append("--face-selector-gender " + str(options["face_selector_gender"]))
    if options.get("face_detector_score"):
        parts.append("--face-detector-score " + str(options["face_detector_score"]))
    return " ".join(parts)


def handler(event):
    t0 = time.time()
    inp = event.get("input") or {}

    source_id = inp.get("source_drive_file_id")
    target_filename = inp.get("target_filename")
    output_folder_id = inp.get("output_folder_id")
    refs_folder_id = inp.get("references_folder_id")
    ref_filenames = inp.get("reference_filenames") or []
    options = inp.get("faceswap_options") or {}
    result_webhook_url = inp.get("result_webhook_url")
    order = inp.get("order")
    scene_id = inp.get("scene_id")

    missing = [k for k, v in {
        "source_drive_file_id": source_id,
        "target_filename": target_filename,
        "output_folder_id": output_folder_id,
        "references_folder_id": refs_folder_id,
        "result_webhook_url": result_webhook_url,
    }.items() if not v]
    if missing:
        return {"status": "error", "code": "BAD_INPUT",
                "message": "missing: " + ", ".join(missing)}

    try:
        svc = _drive()
    except Exception as e:
        return {"status": "error", "code": "DRIVE_AUTH", "message": str(e)[:300]}

    workdir = tempfile.mkdtemp(prefix="fsjob_")
    refs_dir = os.path.join(workdir, "refs")
    os.makedirs(refs_dir, exist_ok=True)
    target_path = os.path.join(workdir, "in.mp4")
    output_path = os.path.join(workdir, "out.mp4")

    try:
        # 1. Download source + refs (read via service account)
        drive_download(svc, source_id, target_path)
        for fn in ref_filenames:
            found = drive_find(svc, refs_folder_id, fn)
            if not found:
                return {"status": "error", "code": "REF_MISSING",
                        "message": "ref not found: " + fn,
                        "order": order, "scene_id": scene_id}
            drive_download(svc, found[0]["id"], os.path.join(refs_dir, fn))
        if not glob.glob(os.path.join(refs_dir, "*.png")):
            return {"status": "error", "code": "NO_REFS",
                    "message": "no .png refs downloaded",
                    "order": order, "scene_id": scene_id}

        # 2. FaceFusion
        cmd = build_facefusion_command(
            os.path.join(refs_dir, "*.png"), target_path, output_path, options)
        proc = subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
            timeout=PER_SCENE_TIMEOUT_SECONDS)
        if proc.returncode != 0 or not os.path.isfile(output_path):
            tail = (proc.stderr or proc.stdout or "")[-800:]
            return {"status": "error", "code": "FACEFUSION_ERROR", "message": tail,
                    "order": order, "scene_id": scene_id,
                    "runtime_seconds": round(time.time() - t0, 1)}

        # 3. Envoyer le MP4 au webhook n8n (qui uploadera sur Drive via OAuth)
        with open(output_path, "rb") as fh:
            data = fh.read()
        fields = {
            "target_filename": target_filename,
            "output_folder_id": str(output_folder_id),
            "order": str(order),
            "scene_id": str(scene_id or ""),
        }
        try:
            code, resp_text = post_multipart(
                result_webhook_url, fields, "data", target_filename, data)
        except Exception as e:
            return {"status": "error", "code": "WEBHOOK_POST_FAILED",
                    "message": "%s: %s" % (type(e).__name__, str(e)[:300]),
                    "order": order, "scene_id": scene_id,
                    "size_bytes": len(data), "runtime_seconds": round(time.time() - t0, 1)}

        return {
            "status": "ok",
            "filename": target_filename,
            "order": order,
            "scene_id": scene_id,
            "size_bytes": len(data),
            "webhook_http_code": code,
            "webhook_response": resp_text,
            "runtime_seconds": round(time.time() - t0, 1),
        }
    except subprocess.TimeoutExpired:
        return {"status": "error", "code": "TIMEOUT",
                "message": "facefusion exceeded %ds" % PER_SCENE_TIMEOUT_SECONDS,
                "order": order, "scene_id": scene_id}
    except Exception as e:
        return {"status": "error", "code": "INTERNAL",
                "message": "%s: %s" % (type(e).__name__, str(e)[:300]),
                "order": order, "scene_id": scene_id}
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


runpod.serverless.start({"handler": handler})
