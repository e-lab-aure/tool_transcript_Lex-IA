import functools
import math
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import traceback
import uuid
from datetime import datetime

import numpy as np
import soundfile as sf
import torch
from flask import Flask, request, jsonify, send_from_directory, send_file

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

app = Flask(__name__, static_folder="static")

# ---------------------------------------------------------------------------
# Binaires
# ---------------------------------------------------------------------------
_SCRIPTS  = os.path.dirname(sys.executable)
_YTDLP_EXE = os.path.join(_SCRIPTS, "yt-dlp.exe")
YTDLP_BIN = _YTDLP_EXE if os.path.isfile(_YTDLP_EXE) else "yt-dlp"
FFMPEG_BIN = "ffmpeg"

# ---------------------------------------------------------------------------
# Modeles
# ---------------------------------------------------------------------------
print("⏳ Chargement de Whisper large-v3 (faster-whisper)...")
from faster_whisper import WhisperModel
whisper_model = WhisperModel("large-v3", device="cuda", compute_type="float16")
print("✅ Whisper large-v3 pret.")

print("⏳ Chargement de Canary-Qwen-2.5B (NeMo)...")
from nemo.collections.speechlm2.models import SALM
canary_model = SALM.from_pretrained("nvidia/canary-qwen-2.5b").bfloat16().eval().to("cuda")
print("✅ Canary-Qwen-2.5B pret.")

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
CANARY_CHUNK_SECS = 40
SAMPLE_RATE       = 16000

SUPPORTED_LANGUAGES = {
    "auto": "Détection automatique",
    "en":   "Anglais",
    "fr":   "Français",
    "de":   "Allemand",
    "es":   "Espagnol",
    "it":   "Italien",
    "pt":   "Portugais",
    "nl":   "Néerlandais",
    "pl":   "Polonais",
    "ru":   "Russe",
    "zh":   "Chinois",
    "ja":   "Japonais",
    "ko":   "Coréen",
    "ar":   "Arabe",
}

# ---------------------------------------------------------------------------
# Store des jobs
# ---------------------------------------------------------------------------
jobs: dict        = {}
_audio_files: dict = {}  # job_id -> chemin MP3 persistant


def _new_job(title: str | None = None) -> dict:
    return {
        "status":     "queued",
        "title":      title,
        "transcript": None,
        "language":   None,
        "model_used": None,
        "diarized":        False,
        "audio_available": False,
        "test_mode":       False,
        "error":           None,
        "logs":            [],
    }


def log(job_id: str, msg: str, level: str = "info") -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    jobs[job_id]["logs"].append({"ts": ts, "msg": msg, "level": level})
    print(f"[{ts}] [{level.upper()}] {msg}")


def _fail_job(job_id: str, msg: str) -> None:
    log(job_id, msg, "error")
    jobs[job_id]["status"] = "error"
    jobs[job_id]["error"]  = msg


def _finish_job(job_id: str, transcript: str, detected_lang: str, model_used: str) -> None:
    log(job_id, f"Transcription terminée : {len(transcript.split())} mots", "success")
    log(job_id, f"Modèle utilisé : {model_used}", "info")
    jobs[job_id].update({
        "status":     "done",
        "transcript": transcript,
        "language":   detected_lang,
        "model_used": model_used,
    })


# ---------------------------------------------------------------------------
# Helpers audio
# ---------------------------------------------------------------------------

def convert_to_wav(src_path: str, dst_path: str) -> None:
    subprocess.run(
        [FFMPEG_BIN, "-y", "-i", src_path, "-ar", str(SAMPLE_RATE), "-ac", "1", dst_path],
        capture_output=True,
        check=True,
    )


def split_audio(wav_path: str, chunk_secs: int, tmpdir: str) -> tuple[list, float]:
    audio, sr = sf.read(wav_path, dtype="float32")
    assert sr == SAMPLE_RATE
    total_secs    = len(audio) / sr
    chunk_samples = chunk_secs * sr
    chunk_paths   = []
    for i in range(math.ceil(len(audio) / chunk_samples)):
        start = i * chunk_samples
        chunk = audio[start : min(start + chunk_samples, len(audio))]
        path  = os.path.join(tmpdir, f"chunk_{i:04d}.wav")
        sf.write(path, chunk, sr)
        chunk_paths.append(path)
    return chunk_paths, total_secs


def preprocess_audio(wav_path: str, job_id: str | None = None) -> None:
    """
    Chaine DSP en place : filtre HP 80 Hz, passe-bande 100-8 kHz,
    reduction de bruit spectral, normalisation RMS -20 dBFS, limiteur soft.
    """
    from scipy.signal import butter, sosfilt
    import noisereduce as nr

    def _soft_limit(x: np.ndarray, threshold: float) -> np.ndarray:
        gain = np.where(
            np.abs(x) > threshold,
            threshold / np.maximum(np.abs(x), 1e-9) * np.tanh(np.abs(x) / threshold),
            1.0,
        )
        return x * gain

    if job_id:
        log(job_id, "Prétraitement audio (filtre HP, débruitage, normalisation)...", "step")

    audio, sr = sf.read(wav_path, dtype="float32")
    if len(audio) < 2048:
        return

    # Filtre passe-haut 80 Hz — supprime vent / bourdonnement 50-60 Hz
    audio = sosfilt(butter(4, 80.0, btype="high", fs=sr, output="sos"), audio)

    # Passe-bande 100-8000 Hz — shaping doux sur la plage voix
    nyq   = sr / 2.0
    audio = sosfilt(butter(2, [100.0, min(7999.0, nyq - 1)], btype="bandpass", fs=sr, output="sos"), audio)

    # Reduction de bruit spectral stationnaire (spectral gating, 75%)
    audio = nr.reduce_noise(y=audio, sr=sr, stationary=True, prop_decrease=0.75, n_fft=1024, n_jobs=1)

    # Normalisation RMS vers -20 dBFS
    rms = np.sqrt(np.mean(audio ** 2))
    if rms > 1e-9:
        audio = audio * (10 ** (-20.0 / 20.0) / rms)

    # Limiteur soft tanh
    audio = _soft_limit(audio, threshold=0.95)

    sf.write(wav_path, audio, sr, subtype="PCM_16")
    if job_id:
        log(job_id, "Prétraitement audio terminé.", "success")


# ---------------------------------------------------------------------------
# Diarisation
# ---------------------------------------------------------------------------

_diarization_pipeline = None


def _get_diarization_pipeline():
    global _diarization_pipeline
    if _diarization_pipeline is not None:
        return _diarization_pipeline

    import huggingface_hub

    # pyannote 3.x appelle hf_hub_download(use_auth_token=...) supprime dans
    # huggingface_hub >= 1.0 — on traduit use_auth_token -> token une seule fois.
    if not getattr(huggingface_hub.hf_hub_download, "_compat_patched", False):
        def _compat(fn):
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                if "use_auth_token" in kwargs:
                    kwargs.setdefault("token", kwargs.pop("use_auth_token"))
                return fn(*args, **kwargs)
            wrapper._compat_patched = True
            return wrapper
        for _name in ("hf_hub_download", "snapshot_download"):
            if hasattr(huggingface_hub, _name):
                setattr(huggingface_hub, _name, _compat(getattr(huggingface_hub, _name)))

    from pyannote.audio import Pipeline

    hf_token = os.environ.get("HF_TOKEN", "").strip()
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN manquant. Créez un token sur huggingface.co, acceptez les licences "
            "de pyannote/speaker-diarization-3.1 et pyannote/segmentation-3.0, "
            "puis définissez HF_TOKEN dans .env."
        )

    huggingface_hub.login(token=hf_token, add_to_git_credential=False)
    _diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
    if torch.cuda.is_available():
        _diarization_pipeline = _diarization_pipeline.to(torch.device("cuda"))
    return _diarization_pipeline


def diarize_audio(wav_path: str, job_id: str | None = None) -> list[dict]:
    """Retourne [{start, end, speaker}, ...] via pyannote/speaker-diarization-3.1."""
    if job_id:
        log(job_id, "Diarisation : identification des locuteurs (pyannote)...", "step")
    diarization = _get_diarization_pipeline()(wav_path)
    segments = [
        {"start": turn.start, "end": turn.end, "speaker": speaker}
        for turn, _, speaker in diarization.itertracks(yield_label=True)
    ]
    if job_id:
        n_speakers = len({s["speaker"] for s in segments})
        log(job_id, f"Diarisation terminée : {n_speakers} locuteur(s) identifié(s)", "success")
    return segments


def assign_speakers(whisper_segments: list, diar_segments: list) -> str:
    """
    Aligne les segments Whisper (timestampés) sur les segments de diarisation
    par chevauchement maximal. Produit : [Locuteur N | mm:ss -> mm:ss] texte
    """
    seen: list = []
    for d in diar_segments:
        if d["speaker"] not in seen:
            seen.append(d["speaker"])
    names = {sp: f"Locuteur {i + 1}" for i, sp in enumerate(seen)}

    lines = []
    for seg in whisper_segments:
        best, best_overlap = None, 0.0
        for d in diar_segments:
            overlap = max(0.0, min(seg.end, d["end"]) - max(seg.start, d["start"]))
            if overlap > best_overlap:
                best_overlap, best = overlap, d["speaker"]
        name = names.get(best, "Inconnu") if best else "Inconnu"
        sm, ss = int(seg.start // 60), seg.start % 60
        em, es = int(seg.end // 60), seg.end % 60
        lines.append(f"[{name} | {sm:02d}:{ss:05.2f} -> {em:02d}:{es:05.2f}] {seg.text.strip()}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

def transcribe_with_canary(wav_path: str, tmpdir: str, job_id: str) -> str:
    log(job_id, "Découpage de l'audio en segments de 40s (limite Canary-Qwen)...", "step")
    chunk_paths, total_secs = split_audio(wav_path, CANARY_CHUNK_SECS, tmpdir)
    n = len(chunk_paths)
    log(job_id, f"Audio total : {total_secs:.1f}s → {n} chunk{'s' if n > 1 else ''} de {CANARY_CHUNK_SECS}s", "info")
    parts = []
    for i, chunk_path in enumerate(chunk_paths):
        start_s = i * CANARY_CHUNK_SECS
        end_s   = min((i + 1) * CANARY_CHUNK_SECS, total_secs)
        log(job_id, f"Canary-Qwen-2.5B — chunk {i+1}/{n} [{start_s:.0f}s → {end_s:.0f}s] en cours...", "model")
        answer_ids = canary_model.generate(
            prompts=[[{
                "role":    "user",
                "content": f"Transcribe the following: {canary_model.audio_locator_tag}",
                "audio":   [chunk_path],
            }]],
            max_new_tokens=1024,
        )
        text = canary_model.tokenizer.ids_to_text(answer_ids[0].cpu()).strip()
        log(job_id, f"Chunk {i+1}/{n} transcrit ({len(text)} caractères)", "success")
        parts.append(text)
    log(job_id, f"Assemblage des {n} chunks en transcript final...", "step")
    return " ".join(parts)


def transcribe_with_whisper(
    wav_path: str,
    language: str | None,
    job_id: str | None = None,
    return_segments: bool = False,
):
    segments_list = []
    segments, info = whisper_model.transcribe(
        wav_path,
        beam_size=5,
        language=language if language not in (None, "auto") else None,
        vad_filter=True,
    )
    for seg in segments:
        segments_list.append(seg)
        if job_id:
            log(job_id, f"[{seg.start:.1f}s → {seg.end:.1f}s] {seg.text.strip()}", "segment")
    transcript = "\n".join(s.text.strip() for s in segments_list)
    if return_segments:
        return transcript, info.language, segments_list
    return transcript, info.language


def _route_and_transcribe(
    wav_path: str,
    tmpdir: str,
    language: str,
    diarize: bool,
    job_id: str,
) -> tuple[str, str, str]:
    """
    Routing vers le bon modele. Retourne (transcript, detected_lang, model_used).
    Quand diarize=True : Whisper uniquement (timestamps requis), puis pyannote.
    """
    if diarize:
        log(job_id, "Diarisation activée — Whisper large-v3 utilisé pour les timestamps...", "step")
        lang_param = None if language == "auto" else language
        transcript, detected_lang, segments = transcribe_with_whisper(
            wav_path, lang_param, job_id, return_segments=True
        )
        transcript = assign_speakers(segments, diarize_audio(wav_path, job_id))
        jobs[job_id]["diarized"] = True
        return transcript, detected_lang, "Whisper large-v3 + Diarisation"

    if language == "en":
        log(job_id, "Langue : anglais forcé → Canary-Qwen-2.5B", "step")
        return transcribe_with_canary(wav_path, tmpdir, job_id), "en", "Canary-Qwen-2.5B"

    if language == "auto":
        log(job_id, "Détection automatique de la langue via Whisper...", "step")
        jobs[job_id]["status"] = "detecting"
        _, detected_lang = transcribe_with_whisper(wav_path, None)
        log(job_id, f"Langue détectée : {detected_lang}", "info")
        jobs[job_id]["status"] = "transcribing"
        if detected_lang == "en":
            log(job_id, "Anglais détecté → Canary-Qwen-2.5B", "step")
            return transcribe_with_canary(wav_path, tmpdir, job_id), "en", "Canary-Qwen-2.5B (anglais détecté)"
        log(job_id, f"Langue non-anglaise ({detected_lang}) → Whisper large-v3", "step")
        transcript, detected_lang = transcribe_with_whisper(wav_path, None, job_id)
        return transcript, detected_lang, f"Whisper large-v3 ({detected_lang} détecté)"

    log(job_id, f"Langue explicite ({language}) → Whisper large-v3", "step")
    transcript, detected_lang = transcribe_with_whisper(wav_path, language, job_id)
    return transcript, detected_lang, "Whisper large-v3"


# ---------------------------------------------------------------------------
# Audio persistant pour téléchargement
# ---------------------------------------------------------------------------

def _trim_wav(wav_path: str, duration_secs: int) -> None:
    """Tronque le WAV en place à duration_secs secondes."""
    tmp = wav_path + ".trim.wav"
    subprocess.run(
        [FFMPEG_BIN, "-y", "-i", wav_path, "-t", str(duration_secs), tmp],
        capture_output=True, check=True,
    )
    os.replace(tmp, wav_path)


def _save_job_audio(job_id: str, wav_path: str) -> None:
    """Re-encode le WAV traité en MP3 128k et le conserve pour téléchargement."""
    mp3_path = os.path.join(tempfile.gettempdir(), f"lexia_{job_id}.mp3")
    try:
        subprocess.run(
            [FFMPEG_BIN, "-y", "-i", wav_path, "-b:a", "128k", mp3_path],
            capture_output=True, check=True,
        )
        _audio_files[job_id] = mp3_path
        jobs[job_id]["audio_available"] = True
        log(job_id, "Audio MP3 disponible pour téléchargement.", "info")
    except subprocess.CalledProcessError as e:
        log(job_id, f"Encodage MP3 échoué (téléchargement indisponible) : {e.stderr.decode()}", "error")


# ---------------------------------------------------------------------------
# Runners (threads)
# ---------------------------------------------------------------------------

def run_transcription_from_file(
    job_id: str,
    file_path: str,
    upload_dir: str,
    language: str,
    preprocess: bool = True,
    diarize: bool = False,
    test_mode: bool = False,
) -> None:
    log(job_id, "Démarrage du job depuis un fichier local", "step")
    log(job_id, f"Fichier : {os.path.basename(file_path)}", "info")
    log(job_id, f"Langue sélectionnée : {SUPPORTED_LANGUAGES.get(language, language)}", "info")

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            size_mb = os.path.getsize(file_path) / 1024 / 1024
            log(job_id, f"Taille du fichier : {size_mb:.1f} Mo", "info")

            log(job_id, "Conversion en WAV 16kHz mono via ffmpeg...", "step")
            wav_path = os.path.join(tmpdir, "audio.wav")
            try:
                convert_to_wav(file_path, wav_path)
            except subprocess.CalledProcessError as e:
                _fail_job(job_id, f"Erreur ffmpeg : {e.stderr.decode()}")
                return
            log(job_id, f"WAV généré : {os.path.getsize(wav_path)/1024/1024:.1f} Mo @ {SAMPLE_RATE}Hz mono", "success")

            if test_mode:
                _trim_wav(wav_path, 15)
                log(job_id, "Mode test : audio tronqué à 15 secondes.", "step")
                jobs[job_id]["test_mode"] = True

            if preprocess:
                preprocess_audio(wav_path, job_id)

            _save_job_audio(job_id, wav_path)

            jobs[job_id]["status"] = "transcribing"
            transcript, detected_lang, model_used = _route_and_transcribe(
                wav_path, tmpdir, language, diarize, job_id
            )
            _finish_job(job_id, transcript, detected_lang, model_used)

    except Exception as e:
        _fail_job(job_id, f"Erreur : {e}\n{traceback.format_exc()}")
    finally:
        shutil.rmtree(upload_dir, ignore_errors=True)


def run_transcription(
    job_id: str,
    url: str,
    language: str,
    preprocess: bool = True,
    diarize: bool = False,
    test_mode: bool = False,
) -> None:
    jobs[job_id]["status"] = "downloading"
    log(job_id, "Démarrage du job de transcription", "step")
    log(job_id, f"URL cible : {url}", "info")
    log(job_id, f"Langue sélectionnée : {SUPPORTED_LANGUAGES.get(language, language)}", "info")

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            log(job_id, "Récupération du titre de la vidéo via yt-dlp...", "step")
            title_result = subprocess.run(
                [YTDLP_BIN, "--no-playlist", "--print", "title", url],
                capture_output=True, text=True,
            )
            title = title_result.stdout.strip().split("\n")[0] or "transcript"
            jobs[job_id]["title"] = title
            log(job_id, f"Titre : « {title} »", "info")

            log(job_id, "Téléchargement de la piste audio (yt-dlp → mp3)...", "step")
            dl = subprocess.run(
                [YTDLP_BIN, "-x", "--audio-format", "mp3", "--audio-quality", "0",
                 "-o", os.path.join(tmpdir, "audio.%(ext)s"), "--no-playlist", url],
                capture_output=True, text=True,
            )
            if dl.returncode != 0:
                _fail_job(job_id, dl.stderr.strip() or "Erreur lors du téléchargement")
                return

            audio_file = next(
                (os.path.join(tmpdir, f) for f in os.listdir(tmpdir)
                 if f.endswith((".mp3", ".m4a", ".webm", ".opus"))),
                None,
            )
            if not audio_file:
                _fail_job(job_id, f"Fichier audio introuvable. Présents : {os.listdir(tmpdir)}")
                return
            log(job_id, f"Audio téléchargé : {os.path.basename(audio_file)} ({os.path.getsize(audio_file)/1024/1024:.1f} Mo)", "success")

            log(job_id, "Conversion en WAV 16kHz mono (requis par les modèles ASR)...", "step")
            wav_path = os.path.join(tmpdir, "audio.wav")
            try:
                convert_to_wav(audio_file, wav_path)
            except subprocess.CalledProcessError as e:
                _fail_job(job_id, f"Erreur ffmpeg : {e.stderr.decode()}")
                return
            log(job_id, f"WAV généré : {os.path.getsize(wav_path)/1024/1024:.1f} Mo @ {SAMPLE_RATE}Hz mono", "success")

            if test_mode:
                _trim_wav(wav_path, 15)
                log(job_id, "Mode test : audio tronqué à 15 secondes.", "step")
                jobs[job_id]["test_mode"] = True

            if preprocess:
                preprocess_audio(wav_path, job_id)

            _save_job_audio(job_id, wav_path)

            jobs[job_id]["status"] = "transcribing"
            transcript, detected_lang, model_used = _route_and_transcribe(
                wav_path, tmpdir, language, diarize, job_id
            )
            _finish_job(job_id, transcript, detected_lang, model_used)

    except Exception as e:
        _fail_job(job_id, f"Erreur : {e}\n{traceback.format_exc()}")


# ---------------------------------------------------------------------------
# Routes Flask
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/languages")
def languages():
    return jsonify(SUPPORTED_LANGUAGES)


@app.route("/transcribe", methods=["POST"])
def transcribe():
    data       = request.get_json() or {}
    url        = data.get("url", "").strip()
    language   = data.get("language", "auto").strip()
    preprocess = bool(data.get("preprocess", True))
    diarize    = bool(data.get("diarize", False))
    test_mode  = bool(data.get("test_mode", False))

    if not url:
        return jsonify({"error": "URL manquante"}), 400
    if language not in SUPPORTED_LANGUAGES:
        return jsonify({"error": f"Langue non supportée : {language}"}), 400

    job_id = str(uuid.uuid4())
    jobs[job_id] = _new_job()
    threading.Thread(
        target=run_transcription,
        args=(job_id, url, language, preprocess, diarize, test_mode),
        daemon=True,
    ).start()
    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job introuvable"}), 404
    return jsonify(job)


@app.route("/logs/<job_id>")
def get_logs(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job introuvable"}), 404
    return jsonify({"logs": job["logs"]})


@app.route("/test-diarize")
def test_diarize():
    try:
        _get_diarization_pipeline()
        return jsonify({"ok": True, "message": "Pipeline pyannote chargé — connexion et token valides."})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/download-audio/<job_id>")
def download_audio(job_id):
    path = _audio_files.get(job_id)
    if not path or not os.path.isfile(path):
        return jsonify({"error": "Fichier audio non disponible"}), 404
    title = (jobs.get(job_id) or {}).get("title") or "audio"
    safe  = "".join(c if c.isalnum() or c in "- _" else "_" for c in title)
    return send_file(path, as_attachment=True, download_name=f"{safe}.mp3", mimetype="audio/mpeg")


@app.route("/transcribe-file", methods=["POST"])
def transcribe_file():
    if "file" not in request.files:
        return jsonify({"error": "Aucun fichier reçu"}), 400

    f          = request.files["file"]
    language   = request.form.get("language", "auto").strip()
    preprocess = request.form.get("preprocess", "true").lower() == "true"
    diarize    = request.form.get("diarize", "false").lower() == "true"
    test_mode  = request.form.get("test_mode", "false").lower() == "true"

    if not f.filename:
        return jsonify({"error": "Nom de fichier invalide"}), 400
    if language not in SUPPORTED_LANGUAGES:
        return jsonify({"error": f"Langue non supportée : {language}"}), 400

    upload_dir = tempfile.mkdtemp(prefix="upload_")
    ext        = os.path.splitext(f.filename)[1].lower() or ".bin"
    safe_path  = os.path.join(upload_dir, f"input{ext}")
    f.save(safe_path)

    title  = os.path.splitext(f.filename)[0]
    job_id = str(uuid.uuid4())
    jobs[job_id] = _new_job(title)
    threading.Thread(
        target=run_transcription_from_file,
        args=(job_id, safe_path, upload_dir, language, preprocess, diarize, test_mode),
        daemon=True,
    ).start()
    return jsonify({"job_id": job_id})


if __name__ == "__main__":
    app.run(debug=False, port=5000, threaded=True)
