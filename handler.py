#!/usr/bin/env python3
"""
RunPod Serverless handler — Face Swap (1 scène = 1 job).

Reproduit À L'IDENTIQUE la logique de faceswap_batch.py (build_facefusion_command),
mais SANS création de pod / SSH / install : FaceFusion est déjà cuit dans l'image.

Flux d'un job :
  1. (idempotence) si target_filename existe déjà dans output_folder_id -> skip
  2. download la source MP4 depuis Drive (source_drive_file_id)
  3. download les refs PNG depuis references_folder_id
  4. lance FaceFusion headless-run (mêmes args qu'aujourd'hui)
  5. upload le MP4 résultat dans output_folder_id (Service Account Drive)
  6. retourne { status, drive_file_id, filename, runtime_seconds }

Entrée (event["input"]) :
{
  "source_drive_file_id": "1AbC...",
  "target_filename":      "scene_007_faceswap.mp4",
  "output_folder_id":     "1Out...",        # 07_faceswap_segments
  "references_folder_id":  "1Ref...",       # face_swap_keyframes
  "reference_filenames":  ["ref_01.png", "ref_02.png", ...],
  "faceswap_options":     { ... },          # repris tel quel du face_swap_plan
  "scene_id":             "sc_07",          # optionnel, pour le log
  "order":                7                  # optionnel, pour le log
}

Variables d'environnement requises sur l'endpoint :
  GOOGLE_SERVICE_ACCOUNT_JSON   le contenu JSON du service account (sa-drive.json)
  FACEFUSION_WORKDIR            défaut /app
"""

import os
import io
import json
import time
import glob
import shutil
import tempfile
import subprocess

import runpod
from google.oauth2 import service_account
from googleapiclient.discovery import build as gbuild
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload

FACEFUSION_WORKDIR = os.environ.get("FACEFUSION_WORKDIR", "/app")
SWAPPER_MODEL_DEFAULT = os.environ.get("FACESWAP_SWAPPER_MODEL", "inswapper_128")
SA_SCOPE = ["https://www.googleapis.com/auth/drive"]
PER_SCENE_TIMEOUT_SECONDS = int(os.environ.get("PER_SCENE_TIMEOUT_SECONDS", "1800"))


# ----------------------------------------------------------------------------
# Google Drive (Service Account) — mêmes opérations que faceswap_batch.py
# ----------------------------------------------------------------------------
def _drive():
    raw = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not raw:
        raise RuntimeError("GOOGLE_SERVICE_ACCOUNT_JSON env var missing")
    info = json.loads(raw)
    creds = service_account.Credentials.from_service_account_info(info, scopes=SA_SCOPE)
    return gbuild("drive", "v3", credentials=creds, cache_discovery=False)


def drive_find(svc, parent_id, name):
    q = (
        "name = '" + name.replace("'", "\\'") + "' and '"
        + parent_id + "' in parents and trashed = false"
    )
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


def drive_upload(svc, parent_id, name, src_path, mime="video/mp4"):
    meta = {"name": name, "parents": [parent_id]}
    media = MediaFileUpload(src_path, mimetype=mime, resumable=True)
    f = svc.files().create(
        body=meta, media_body=media, fields="id,name",
        supportsAllDrives=True,
    ).execute()
    return f


# ----------------------------------------------------------------------------
# Commande FaceFusion — PORTÉE À L'IDENTIQUE de build_facefusion_command()
# ----------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------
# Handler RunPod
# ----------------------------------------------------------------------------
def handler(event):
    t0 = time.time()
    inp = event.get("input") or {}

    source_id = inp.get("source_drive_file_id")
    target_filename = inp.get("target_filename")
    output_folder_id = inp.get("output_folder_id")
    refs_folder_id = inp.get("references_folder_id")
    ref_filenames = inp.get("reference_filenames") or []
    options = inp.get("faceswap_options") or {}
    order = inp.get("order")
    scene_id = inp.get("scene_id")

    missing = [k for k, v in {
        "source_drive_file_id": source_id,
        "target_filename": target_filename,
        "output_folder_id": output_folder_id,
        "references_folder_id": refs_folder_id,
    }.items() if not v]
    if missing:
        return {"status": "error", "code": "BAD_INPUT",
                "message": "missing: " + ", ".join(missing)}

    try:
        svc = _drive()
    except Exception as e:
        return {"status": "error", "code": "DRIVE_AUTH", "message": str(e)[:300]}

    # 1. Idempotence — si la sortie existe déjà, on ne refait pas (et $0)
    try:
        existing = drive_find(svc, output_folder_id, target_filename)
        if existing:
            return {
                "status": "ok", "skipped": True, "code": "OUTPUT_EXISTS",
                "drive_file_id": existing[0]["id"], "filename": target_filename,
                "order": order, "scene_id": scene_id,
                "runtime_seconds": round(time.time() - t0, 1),
            }
    except Exception as e:
        return {"status": "error", "code": "DRIVE_FIND", "message": str(e)[:300]}

    workdir = tempfile.mkdtemp(prefix="fsjob_")
    refs_dir = os.path.join(workdir, "refs")
    os.makedirs(refs_dir, exist_ok=True)
    target_path = os.path.join(workdir, "in.mp4")
    output_path = os.path.join(workdir, "out.mp4")

    try:
        # 2. Download source
        drive_download(svc, source_id, target_path)

        # 3. Download refs (par nom dans le dossier refs)
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

        # 4. FaceFusion
        cmd = build_facefusion_command(
            refs_glob=os.path.join(refs_dir, "*.png"),
            target=target_path,
            output=output_path,
            options=options,
        )
        proc = subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
            timeout=PER_SCENE_TIMEOUT_SECONDS,
        )
        if proc.returncode != 0 or not os.path.isfile(output_path):
            tail = (proc.stderr or proc.stdout or "")[-800:]
            return {"status": "error", "code": "FACEFUSION_ERROR",
                    "message": tail, "order": order, "scene_id": scene_id,
                    "runtime_seconds": round(time.time() - t0, 1)}

        # 5. Upload résultat sur Drive
        up = drive_upload(svc, output_folder_id, target_filename, output_path)

        return {
            "status": "ok", "skipped": False,
            "drive_file_id": up["id"], "filename": up["name"],
            "order": order, "scene_id": scene_id,
            "size_bytes": os.path.getsize(output_path),
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
