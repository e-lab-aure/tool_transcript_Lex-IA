import functools
import math
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import warnings
from collections import deque
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from queue import Queue
from urllib.parse import urlsplit

import numpy as np
import soundfile as sf
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_from_directory, send_file
from werkzeug.exceptions import HTTPException

load_dotenv(Path(__file__).with_name(".env"))

app = Flask(__name__, static_folder=None)
STATIC_DIR = Path(__file__).with_name("static")
app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024
LIVE_UPLOAD_LIMIT = 16 * 1024 * 1024
LIVE_MAX_SECONDS = 120
MAX_AUDIO_SECONDS = 4 * 3600
MEDIA_TIMEOUT = 300
DOWNLOAD_TIMEOUT = 1800

SERVER_PORT = 5000


def ensure_port_free(port: int) -> None:
    """
    Evite de charger les modeles si un serveur ecoute deja sur le port.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        if sock.connect_ex(("127.0.0.1", port)) == 0:
            print(f"❌ Le port {port} est déjà utilisé : un serveur LEX-IA tourne probablement déjà "
                  "(icône de notification ou autre console). Arrêtez-le avant de relancer.")
            sys.exit(1)


# ---------------------------------------------------------------------------
# Binaires
# ---------------------------------------------------------------------------
YTDLP_BIN = [sys.executable, "-m", "yt_dlp", "--ignore-config", "--no-playlist",
             "--socket-timeout", "30", "--retries", "3"]
YTDLP_ENV = {**os.environ, "PYTHONIOENCODING": "utf-8",
             "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")}
FFMPEG_BIN = "ffmpeg"

# ---------------------------------------------------------------------------
# Modeles
# ---------------------------------------------------------------------------
gpu_lock = threading.Lock()


@functools.cache
def _whisper_model():
    from faster_whisper import WhisperModel

    return WhisperModel("large-v3", device="cuda", compute_type="float16")


@functools.cache
def _canary_model():
    from nemo.collections.speechlm2.models import SALM

    return SALM.from_pretrained("nvidia/canary-qwen-2.5b").bfloat16().eval().to("cuda")

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
jobs: dict          = {}
_audio_files: dict  = {}  # job_id -> chemin MP3 persistant
_speaker_clips: dict = {}  # job_id -> {speaker_id: chemin MP3 extrait}
_video_files: dict  = {}  # job_id -> chemin de la video telechargee
jobs_lock = threading.Lock()
finished_at: dict = {}
JOB_TTL = 24 * 3600
MAX_LOGS = 256
JOB_ROOT = Path(tempfile.gettempdir()) / "LEX-IA-jobs"
transcription_slots = threading.BoundedSemaphore(4)
video_slots = threading.BoundedSemaphore(2)
live_slot = threading.Lock()
transcription_queue = Queue()
worker_lock = threading.Lock()
worker_started = False

VIDEO_QUALITIES = {"best": None, "1080": 1080, "720": 720, "480": 480}


def _job_directory(job_id: str) -> Path:
    return JOB_ROOT / str(uuid.UUID(job_id))


def _remove_job_directory(job_id: str) -> None:
    try:
        path = _job_directory(job_id)
        if path.resolve().parent == JOB_ROOT.resolve():
            shutil.rmtree(path)
    except FileNotFoundError:
        pass
    except OSError:
        app.logger.warning("Nettoyage impossible pour le job %s", job_id)


def _purge_jobs() -> None:
    now = time.time()
    with jobs_lock:
        expired = [job_id for job_id, end in finished_at.items() if now - end >= JOB_TTL]
        for job_id in expired:
            for store in (jobs, finished_at, _audio_files, _speaker_clips, _video_files):
                store.pop(job_id, None)
        known = set(jobs)
    for job_id in expired:
        _remove_job_directory(job_id)
    # Reprendre aussi les fichiers abandonnes par un arret force.
    for path in JOB_ROOT.iterdir():
        if path.name in known:
            continue
        try:
            if str(uuid.UUID(path.name)) == path.name and now - path.stat().st_mtime >= JOB_TTL:
                _remove_job_directory(path.name)
        except (ValueError, FileNotFoundError):
            continue


def _cleanup_loop() -> None:
    while True:
        try:
            _purge_jobs()
        except OSError:
            app.logger.exception("Impossible de nettoyer les fichiers de travail")
        time.sleep(300)


def _execute_job(job_id, target, args, slots) -> None:
    try:
        target(job_id, *args)
    except subprocess.TimeoutExpired:
        _fail_job(job_id, "Le téléchargement ou la conversion a dépassé le délai maximal.")
    except Exception:
        app.logger.exception("Echec du job %s", job_id)
        _fail_job(job_id, "Le traitement a échoué. Consultez le journal du serveur.")
    finally:
        if jobs[job_id]["status"] == "error":
            _remove_job_directory(job_id)
        try:
            os.utime(_job_directory(job_id))
        except FileNotFoundError:
            pass
        except OSError:
            app.logger.warning("Horodatage impossible pour le job %s", job_id)
        with jobs_lock:
            finished_at[job_id] = time.time()
        slots.release()


def _transcription_worker() -> None:
    while True:
        task = transcription_queue.get()
        _execute_job(*task)
        transcription_queue.task_done()


def _submit_job(job_id, job, target, args, slots) -> None:
    global worker_started
    with jobs_lock:
        jobs[job_id] = job
    task = (job_id, target, args, slots)
    try:
        if slots is transcription_slots:
            with worker_lock:
                if not worker_started:
                    threading.Thread(target=_transcription_worker, daemon=True).start()
                    worker_started = True
            transcription_queue.put(task)
        else:
            threading.Thread(target=_execute_job, args=task, daemon=True).start()
    except Exception:
        with jobs_lock:
            jobs.pop(job_id, None)
        raise


def _new_job(title: str | None = None, kind: str = "transcription") -> dict:
    return {
        "kind":       kind,
        "status":     "queued",
        "title":      title,
        "transcript": None,
        "language":   None,
        "model_used": None,
        "diarized":        False,
        "diarization_warning": None,
        "rediarize_available": False,
        "speaker_options": {},
        "diarization_turns": [],
        "audio_available": False,
        "test_mode":       0,
        "speakers":        [],
        "segments":        [],
        "duration_secs":   None,
        "error":           None,
        "logs":            deque(maxlen=MAX_LOGS),
        "log_count":       0,
        "processed_secs":  None,
    }


def log(job_id: str, msg: str, level: str = "info") -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    with jobs_lock:
        job = jobs[job_id]
        job["logs"].append({"ts": ts, "msg": msg, "level": level})
        job["log_count"] += 1
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

def convert_to_wav(src_path: str, dst_path: str, limit: int) -> None:
    subprocess.run(
        [FFMPEG_BIN, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
         "-protocol_whitelist", "file,pipe",
         "-format_whitelist", "wav,mp3,flac,ogg,aac,mov,matroska,webm,avi,asf,mpeg,mpegts,aiff",
         "-i", src_path, "-map", "0:a:0",
         "-t", str(limit), "-ar", str(SAMPLE_RATE), "-ac", "1", dst_path],
        capture_output=True,
        check=True,
        timeout=MEDIA_TIMEOUT,
    )


def split_audio(wav_path: str, chunk_secs: int, tmpdir: str) -> tuple[list, float]:
    chunk_paths = []
    with sf.SoundFile(wav_path) as audio:
        sr = audio.samplerate
        if sr != SAMPLE_RATE:
            raise ValueError("Fréquence audio inattendue")
        total_secs = len(audio) / sr
        for i, chunk in enumerate(audio.blocks(blocksize=chunk_secs * sr, dtype="float32")):
            path = os.path.join(tmpdir, f"chunk_{i:04d}.wav")
            sf.write(path, chunk, sr)
            chunk_paths.append(path)
    return chunk_paths, total_secs


def preprocess_audio(wav_path: str, job_id: str | None = None) -> None:
    """
    Chaine DSP en place : passe-bande 100-8 kHz, reduction de bruit,
    normalisation RMS -20 dBFS et limitation des cretes.
    """
    from scipy.signal import butter, sosfilt
    import noisereduce as nr

    if job_id:
        log(job_id, "Prétraitement audio (filtre, débruitage, normalisation)...", "step")

    audio, sr = sf.read(wav_path, dtype="float32")
    if len(audio) < 2048:
        return

    # Passe-bande 100-8000 Hz — shaping doux sur la plage voix
    nyq   = sr / 2.0
    audio = sosfilt(butter(2, [100.0, min(7999.0, nyq - 1)], btype="bandpass", fs=sr, output="sos"), audio)

    # Reduction de bruit spectral stationnaire (spectral gating, 75%)
    audio = nr.reduce_noise(y=audio, sr=sr, stationary=True, prop_decrease=0.75, n_fft=1024, n_jobs=1)

    # Normalisation RMS vers -20 dBFS
    rms = np.sqrt(np.mean(audio ** 2))
    if rms > 1e-9:
        audio = audio * (10 ** (-20.0 / 20.0) / rms)

    peak = np.max(np.abs(audio))
    if peak > 0.95:
        audio *= 0.95 / peak

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

    import torch

    # LEX-IA fournit le WAV sous forme de tenseur, sans le decodeur TorchCodec.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"\s*torchcodec is not installed correctly.*")
        from pyannote.audio import Pipeline
        from pyannote.audio.telemetry.metrics import set_telemetry_metrics

    hf_token = os.environ.get("HF_TOKEN", "").strip()
    if not hf_token:
        raise RuntimeError("HF_TOKEN manquant. Acceptez les conditions de Community-1 sur Hugging Face.")
    set_telemetry_metrics(False)
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1", token=hf_token,
        revision="3533c8cf8e369892e6b79ff1bf80f7b0286a54ee",
    )
    if pipeline is None:
        raise RuntimeError("Impossible de charger Community-1. Vérifiez l'accès au modèle.")
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
    _diarization_pipeline = pipeline
    return pipeline


def diarize_audio(wav_path: str, job_id: str, speaker_options: dict) -> list[dict]:
    import torch

    log(job_id, "Diarisation : identification des locuteurs avec Community-1.", "step")
    pipeline = _get_diarization_pipeline()
    frames, sample_rate = sf.read(wav_path, dtype="float32", always_2d=True)
    output = pipeline(
        {"waveform": torch.from_numpy(frames.T), "sample_rate": sample_rate}, **speaker_options,
    )
    segments = [
        {"start": turn.start, "end": turn.end, "speaker": speaker}
        for turn, _, speaker in output.exclusive_speaker_diarization.itertracks(yield_label=True)
    ]
    count = len({segment["speaker"] for segment in segments})
    log(job_id, f"Diarisation terminée : {count} locuteur(s) identifié(s).", "success")
    return segments




def _build_speaker_meta(segments: list, diar_segments: list) -> list[dict]:
    turns_by_speaker = {}
    for turn in diar_segments:
        turns_by_speaker.setdefault(turn["speaker"], []).append(turn)
    meta = []
    for i, (speaker_id, speaker_diar) in enumerate(turns_by_speaker.items(), 1):
        # Choisir le passage le plus long, avec un extrait de 8 s maximum.
        longest = max(speaker_diar, key=lambda d: d["end"] - d["start"])
        clip_start = longest["start"]
        clip_end = min(longest["end"], clip_start + 8.0)

        excerpt = next((
            seg["text"] for seg in segments if seg["speaker_id"] == speaker_id
            and seg["start"] < clip_end and seg["end"] > clip_start
        ), "")

        meta.append({
            "id":         speaker_id,
            "label":      f"Locuteur {i}",
            "clip_start": clip_start,
            "clip_end":   clip_end,
            "excerpt":    excerpt,
            "speech_secs": round(sum(d["end"] - d["start"] for d in speaker_diar), 1),
        })
    return meta


def _speaker_lookup(diar_segments):
    turns = sorted(diar_segments, key=lambda turn: turn["start"])
    speaker_ids = dict.fromkeys(turn["speaker"] for turn in diar_segments)
    labels = {speaker: f"Locuteur {i}" for i, speaker in enumerate(speaker_ids, 1)}
    first_turn = 0

    def lookup(start, end):
        nonlocal first_turn
        while first_turn < len(turns) and turns[first_turn]["end"] <= start:
            first_turn += 1
        overlaps = {}
        for index in range(first_turn, len(turns)):
            turn = turns[index]
            if turn["start"] >= end:
                break
            overlap = min(end, turn["end"]) - max(start, turn["start"])
            if overlap > 0:
                speaker = turn["speaker"]
                overlaps[speaker] = overlaps.get(speaker, 0.0) + overlap
        speaker = max(overlaps, key=overlaps.get) if overlaps else None
        return {"speaker_id": speaker, "speaker": labels.get(speaker, "Inconnu")}

    return lookup


def _build_segments(whisper_segs: list, diar_segments: list | None = None) -> list[dict]:
    if diar_segments is None:
        return [
            {"start": round(seg.start, 3), "end": round(seg.end, 3),
             "text": seg.text.strip(), "speaker": None}
            for seg in whisper_segs
        ]

    lookup = _speaker_lookup(diar_segments)
    result = []
    for seg in whisper_segs:
        group = None
        words = seg.words
        for part in words or [seg]:
            assignment = lookup(part.start, part.end)
            text = part.word if words else part.text
            if group and group["speaker_id"] == assignment["speaker_id"]:
                group["end"] = round(part.end, 3)
                group["text"] += text
            else:
                group = {
                    "start": round(part.start, 3), "end": round(part.end, 3),
                    "text": text, **assignment,
                }
                result.append(group)
    for segment in result:
        segment["text"] = segment["text"].strip()
    return result


def _timestamp(value: float) -> str:
    minutes, centiseconds = divmod(round(value * 100), 6000)
    return f"{minutes:02d}:{centiseconds / 100:05.2f}"


def _diarized_transcript(segments):
    return "\n".join(
        f"[{seg['speaker']} | {_timestamp(seg['start'])} -> {_timestamp(seg['end'])}] {seg['text']}"
        for seg in segments
    )


def _save_speaker_clips(job_id: str, speaker_meta: list, wav_path: str) -> None:
    """Extrait un court extrait MP3 par locuteur depuis wav_path."""
    clips: dict = {}
    for speaker in speaker_meta:
        duration = speaker["clip_end"] - speaker["clip_start"]
        if duration <= 0:
            continue
        mp3_path = os.path.join(
            _job_directory(job_id),
            f"lexia_{job_id}_spk_{speaker['id']}.mp3",
        )
        try:
            subprocess.run(
                [FFMPEG_BIN, "-y",
                 "-ss", str(speaker["clip_start"]),
                 "-t",  str(duration),
                 "-i",  wav_path,
                 "-b:a", "64k", mp3_path],
                capture_output=True, check=True, timeout=30,
            )
            clips[speaker["id"]] = mp3_path
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            app.logger.warning("Extrait de locuteur indisponible pour le job %s", job_id)
    _speaker_clips[job_id] = clips


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

def transcribe_with_canary(wav_path: str, tmpdir: str, job_id: str, model) -> str:
    chunk_paths, total_secs = split_audio(wav_path, CANARY_CHUNK_SECS, tmpdir)
    parts = []
    for i, chunk_path in enumerate(chunk_paths):
        log(job_id, f"Canary - passage {i + 1}/{len(chunk_paths)}", "model")
        answer_ids = model.generate(
            prompts=[[{
                "role": "user",
                "content": f"Transcribe the following: {model.audio_locator_tag}",
                "audio": [chunk_path],
            }]],
            max_new_tokens=1024,
        )
        parts.append(model.tokenizer.ids_to_text(answer_ids[0].cpu()).strip())
        jobs[job_id]["processed_secs"] = min((i + 1) * CANARY_CHUNK_SECS, total_secs)
    return " ".join(parts)


def _whisper_segments(wav_path: str, language: str, diarize: bool = False):
    return _whisper_model().transcribe(
        wav_path, beam_size=5, language=None if language == "auto" else language,
        vad_filter=True, word_timestamps=diarize,
    )


def _route_and_transcribe(wav_path, raw_wav, tmpdir, language, diarize, speaker_options, job_id,
                          timestamps=False):
    stream = info = None
    if not diarize and not timestamps and language == "auto":
        jobs[job_id]["status"] = "detecting"
        # Whisper fournit la langue avant de consommer le generateur de segments.
        stream, info = _whisper_segments(wav_path, language)
    jobs[job_id]["status"] = "transcribing"
    use_canary = language == "en" or (
        info is not None and info.language == "en" and info.duration_after_vad > 0
    )
    if not diarize and not timestamps and use_canary:
        try:
            model = _canary_model()
            transcript = transcribe_with_canary(wav_path, tmpdir, job_id, model)
        except Exception:
            app.logger.exception("Transcription Canary indisponible")
            log(job_id, "Canary indisponible, transcription avec Whisper.", "info")
            jobs[job_id]["processed_secs"] = 0
        else:
            return transcript, "en", "Canary-Qwen-2.5B"

    if stream is None:
        stream, info = _whisper_segments(wav_path, language, diarize)
    whisper_segments = []
    for segment in stream:
        whisper_segments.append(segment)
        jobs[job_id]["processed_secs"] = segment.end
    diar_segments = None
    if diarize:
        jobs[job_id]["status"] = "diarizing"
        try:
            diar_segments = diarize_audio(raw_wav, job_id, speaker_options)
        except Exception:
            app.logger.exception("Détection des voix indisponible pour le job %s", job_id)
            jobs[job_id]["diarization_warning"] = (
                "Le texte est prêt, mais les voix n’ont pas pu être distinguées. "
                "Vérifiez la détection dans les réglages avancés, puis relancez-la depuis le lecteur."
            )
            log(job_id, "La transcription est conservée sans attribution des voix.", "info")

    segments = _build_segments(whisper_segments, diar_segments)
    jobs[job_id]["segments"] = segments
    if diar_segments is None:
        return "\n".join(seg["text"] for seg in segments), info.language, "Whisper large-v3"

    speaker_meta = _build_speaker_meta(segments, diar_segments)
    jobs[job_id].update(speakers=speaker_meta, diarized=True,
                        speaker_options=speaker_options, diarization_turns=diar_segments)
    _save_speaker_clips(job_id, speaker_meta, raw_wav)
    return _diarized_transcript(segments), info.language, "Whisper large-v3 + Diarisation"


def _save_job_audio(job_id: str, wav_path: str) -> None:
    mp3_path = _job_directory(job_id) / "audio.mp3"
    try:
        subprocess.run(
            [FFMPEG_BIN, "-nostdin", "-hide_banner", "-loglevel", "error",
             "-y", "-i", wav_path, "-b:a", "128k", mp3_path],
            capture_output=True, check=True, timeout=MEDIA_TIMEOUT,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        app.logger.warning("Encodage MP3 impossible pour le job %s", job_id)
        log(job_id, "Téléchargement MP3 indisponible.", "info")
    else:
        _audio_files[job_id] = mp3_path
        jobs[job_id]["audio_available"] = True


def _transcribe_source(job_id, src, tmpdir, language, preprocess, diarize, test_mode, speaker_options,
                       timestamps):
    job = jobs[job_id]
    job["status"] = "preparing"
    raw_wav = os.path.join(tmpdir, "audio.wav")
    try:
        convert_to_wav(src, raw_wav, test_mode or MAX_AUDIO_SECONDS + 1)
    except subprocess.CalledProcessError:
        _fail_job(job_id, "Format audio non reconnu par FFmpeg.")
        return
    duration = sf.info(raw_wav).duration
    if not 0 < duration <= MAX_AUDIO_SECONDS:
        _fail_job(job_id, "L'audio est vide ou dépasse la durée maximale de 4 heures.")
        return
    job.update(duration_secs=round(duration, 2), test_mode=test_mode)
    wav_path = raw_wav
    if preprocess:
        wav_path = os.path.join(tmpdir, "processed.wav")
        shutil.copyfile(raw_wav, wav_path)
        preprocess_audio(wav_path, job_id)

    job["status"] = "waiting_gpu"
    with gpu_lock:
        transcript, detected_lang, model_used = _route_and_transcribe(
            wav_path, raw_wav, tmpdir, language, diarize, speaker_options, job_id, timestamps
        )
    _save_job_audio(job_id, wav_path)
    if job["segments"]:
        try:
            os.replace(raw_wav, _job_directory(job_id) / "source.wav")
        except OSError:
            app.logger.exception("Audio de relance indisponible pour le job %s", job_id)
        else:
            job["rediarize_available"] = True
    _finish_job(job_id, transcript, detected_lang, model_used)


def run_transcription_from_file(job_id, file_path, options) -> None:
    with tempfile.TemporaryDirectory(dir=_job_directory(job_id)) as tmpdir:
        try:
            _transcribe_source(job_id, file_path, tmpdir, **options)
        finally:
            file_path.unlink(missing_ok=True)


def run_transcription(job_id, url, options) -> None:
    jobs[job_id]["status"] = "downloading"
    log(job_id, "Téléchargement de la piste audio.", "step")
    with tempfile.TemporaryDirectory(dir=_job_directory(job_id)) as tmpdir:
        try:
            result = subprocess.run(
                YTDLP_BIN + [
                    "-f", "ba/b", "--max-filesize", "512M", "--no-simulate",
                    "--print", "before_dl:LEXIA_TITLE %(title)s",
                    "--print", "after_move:LEXIA_FILE %(filepath)s",
                    "-o", os.path.join(tmpdir, "source.%(ext)s"), "--", url,
                ],
                env=YTDLP_ENV, encoding="utf-8", errors="replace", capture_output=True,
                check=True, timeout=DOWNLOAD_TIMEOUT,
            )
        except subprocess.CalledProcessError:
            _fail_job(job_id, "Impossible de télécharger l'audio depuis cette URL.")
            return
        source = None
        for line in result.stdout.splitlines():
            if line.startswith("LEXIA_TITLE "):
                jobs[job_id]["title"] = line.removeprefix("LEXIA_TITLE ")[:200]
            elif line.startswith("LEXIA_FILE "):
                source = Path(line.removeprefix("LEXIA_FILE ")).resolve()
        if (source is None or source.parent != Path(tmpdir).resolve()
                or not source.is_file() or source.stat().st_size > app.config["MAX_CONTENT_LENGTH"]):
            _fail_job(job_id, "Aucun audio téléchargé ou limite de taille dépassée.")
            return
        _transcribe_source(job_id, source, tmpdir, **options)


def run_rediarization(job_id, source, original_segments, speaker_options):
    job = jobs[job_id]
    try:
        job["status"] = "waiting_gpu"
        with gpu_lock:
            job["status"] = "diarizing"
            try:
                turns = diarize_audio(source, job_id, speaker_options)
            except Exception:
                app.logger.exception("Détection des voix indisponible pour le job %s", job_id)
                _fail_job(job_id, "La détection des voix a échoué. Vérifiez-la dans les réglages avancés de la page principale.")
                return
        lookup = _speaker_lookup(turns)
        segments = [{**seg, **lookup(seg["start"], seg["end"])} for seg in original_segments]
        speakers = _build_speaker_meta(segments, turns)
        _save_speaker_clips(job_id, speakers, source)
        job.update(segments=segments, speakers=speakers, diarized=True, transcript=_diarized_transcript(segments),
                   diarization_turns=turns, status="done")
    finally:
        source.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Telechargement video (sans transcription)
# ---------------------------------------------------------------------------

# Ligne emise par yt-dlp via --progress-template, parsee pour la barre de progression
_PROGRESS_TAG = "LEXIA_PROGRESS"
_PROGRESS_TEMPLATE = (
    f"download:{_PROGRESS_TAG} %(progress.downloaded_bytes)s|%(progress.total_bytes)s"
    "|%(progress.total_bytes_estimate)s|%(progress.speed)s|%(progress.eta)s"
)


def _video_format_args(quality: str) -> list[str]:
    height = VIDEO_QUALITIES[quality]
    cap = f"[height<={height}]" if height else ""
    return ["-f", f"bv*{cap}+ba/b{cap}", "-S", "res,vcodec:h264,ext:mp4:m4a",
            "--merge-output-format", "mp4"]


def _parse_progress(line: str) -> dict | None:
    """Convertit une ligne LEXIA_PROGRESS en {percent, speed, eta} (valeurs NA -> None)."""
    fields = line.split(_PROGRESS_TAG, 1)[1].strip().split("|")
    if len(fields) != 5:
        return None

    def num(value: str) -> float | None:
        try:
            number = float(value)
            return number if math.isfinite(number) and number >= 0 else None
        except ValueError:
            return None

    downloaded, total, estimate, speed, eta = (num(v) for v in fields)
    total = total or estimate
    if downloaded is None or not total:
        return None
    return {
        "percent": round(min(100.0, downloaded / total * 100), 1),
        "speed":   speed,
        "eta":     int(eta) if eta is not None else None,
    }


def _stop_download(proc) -> None:
    import psutil

    if proc.poll() is not None:
        return
    try:
        parent = psutil.Process(proc.pid)
        processes = parent.children(recursive=True) + [parent]
    except psutil.NoSuchProcess:
        return
    for process in reversed(processes):
        try:
            process.kill()
        except psutil.NoSuchProcess:
            pass
        except psutil.AccessDenied:
            app.logger.warning("Impossible d'arrêter un processus du téléchargement")


def run_video_download(job_id: str, url: str, quality: str) -> None:
    job = jobs[job_id]
    job["status"] = "downloading"
    log(job_id, "Téléchargement des flux vidéo et audio.", "step")
    out_dir = _job_directory(job_id)
    cmd = YTDLP_BIN + _video_format_args(quality) + [
        "--newline", "--no-colors", "--progress", "--no-simulate", "--max-filesize", "2G",
        "--progress-template", _PROGRESS_TEMPLATE,
        "--print", "before_dl:LEXIA_TITLE %(title)s",
        "--print", "after_move:LEXIA_FILE %(filepath)s",
        "-o", str(out_dir / "video.%(ext)s"), "--", url,
    ]
    video_path = None
    stream = 0
    with subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
        text=True, encoding="utf-8", errors="replace", env=YTDLP_ENV,
    ) as proc:
        deadline = threading.Timer(DOWNLOAD_TIMEOUT, _stop_download, args=(proc,))
        deadline.daemon = True
        deadline.start()
        try:
            for raw in proc.stdout:
                line = raw.strip()
                if line.startswith(_PROGRESS_TAG):
                    progress = _parse_progress(line)
                    if progress:
                        job["progress"] = {**progress, "stream": stream}
                elif line.startswith("LEXIA_TITLE "):
                    job["title"] = line.removeprefix("LEXIA_TITLE ")[:200]
                elif line.startswith("LEXIA_FILE "):
                    video_path = Path(line.removeprefix("LEXIA_FILE ")).resolve()
                elif line.startswith("[download] Destination:"):
                    stream += 1
                elif line.startswith("[Merger]"):
                    job["status"] = "merging"
            proc.wait()
        except Exception:
            _stop_download(proc)
            raise
        finally:
            deadline.cancel()

    if (proc.returncode != 0 or video_path is None
            or video_path.parent != out_dir.resolve() or not video_path.is_file()):
        _fail_job(job_id, "Téléchargement vidéo impossible. Vérifiez l'URL, la qualité ou la taille.")
        return
    size = video_path.stat().st_size
    if size > 2 * 1024 ** 3:
        _fail_job(job_id, "La vidéo dépasse la taille maximale de 2 Gio.")
        return

    _video_files[job_id] = video_path
    log(job_id, "Vidéo prête.", "success")
    job.update(
        status="done", video_available=True, video_ext=video_path.suffix.lstrip("."),
        file_size_mb=round(size / 1024 ** 2, 1),
        progress={"percent": 100.0, "speed": None, "eta": 0, "stream": stream},
    )


# ---------------------------------------------------------------------------
# Routes Flask
# ---------------------------------------------------------------------------

@app.errorhandler(HTTPException)
def http_error(exc):
    return jsonify({"error": exc.description}), exc.code


@app.after_request
def live_cors(response):
    if request.endpoint == "transcribe_live":
        response.headers["Access-Control-Allow-Origin"] = "*"
    if request.endpoint in ("transcribe_live", "status", "get_logs", "download_audio",
                            "download_video_file", "speaker_audio"):
        response.headers["Cache-Control"] = "no-store"
    return response


def _json_object():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise ValueError("Un objet JSON est attendu.")
    return data


def _read_url(data):
    value = data.get("url")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("URL manquante.")
    value = value.strip()
    try:
        url = urlsplit(value)
        if (len(value) <= 4096 and url.scheme in ("https", "http") and url.hostname
                and url.username is None and not any(ch.isspace() or ord(ch) < 32 for ch in value)):
            _ = url.port
            return value
    except ValueError:
        pass
    raise ValueError("URL HTTP ou HTTPS invalide.")


def _read_bool(data, key, default, form):
    value = data.get(key, default)
    if type(value) is bool:
        return value
    if form and value.lower() in ("true", "false"):
        return value.lower() == "true"
    raise ValueError(f"{key} doit être un booléen.")


def _read_int(data, key, default, form, low, high):
    value = data.get(key, default)
    if form and isinstance(value, str):
        if value == "":
            return default
        if value.isascii() and value.isdecimal() and len(value) <= 5:
            value = int(value)
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{key} doit être un entier de {low} à {high}.")
    return value


def _read_language(data):
    value = data.get("language", "auto")
    if isinstance(value, str) and value.strip() in SUPPORTED_LANGUAGES:
        return value.strip()
    raise ValueError("Langue non supportée.")


def _read_speaker_options(data, form=False):
    speakers = {}
    for key in ("num_speakers", "min_speakers", "max_speakers"):
        if key in data and data[key] != "":
            speakers[key] = _read_int(data, key, None, form, 1, 20)
    if "num_speakers" in speakers and len(speakers) > 1:
        raise ValueError("Choisissez un nombre exact ou une fourchette de locuteurs.")
    if speakers.get("min_speakers", 1) > speakers.get("max_speakers", 20):
        raise ValueError("Le minimum de locuteurs dépasse le maximum.")
    return speakers


def _read_options(data, form=False):
    diarize = _read_bool(data, "diarize", False, form)
    speakers = _read_speaker_options(data, form)
    if speakers and not diarize:
        raise ValueError("Le nombre de locuteurs nécessite la diarisation.")
    return {
        "language": _read_language(data),
        "preprocess": _read_bool(data, "preprocess", True, form),
        "diarize": diarize,
        "test_mode": _read_int(data, "test_mode", 0, form, 0, MAX_AUDIO_SECONDS),
        "timestamps": _read_bool(data, "timestamps", False, form),
        "speaker_options": speakers,
    }


def _start_job(target, args, *, slots, job=None, upload=None, source_audio=None):
    if not slots.acquire(blocking=False):
        return jsonify({"error": "Trop de traitements en cours. Réessayez dans un instant."}), 429
    job_id = str(uuid.uuid4())
    directory = _job_directory(job_id)
    try:
        directory.mkdir(parents=True)
        if upload is not None:
            source = directory / "input"
            upload.save(source)
            args = (source, *args)
        elif source_audio is not None:
            source = directory / "source.wav"
            # Le travail peut continuer même si le résultat d'origine expire dans la file.
            try:
                os.link(source_audio, source)
            except OSError:
                shutil.copyfile(source_audio, source)
            args = (source, *args)
        _submit_job(job_id, job or _new_job(), target, args, slots)
    except FileNotFoundError:
        slots.release()
        _remove_job_directory(job_id)
        return jsonify({"error": "Audio source indisponible ou expiré."}), 404
    except Exception:
        slots.release()
        _remove_job_directory(job_id)
        app.logger.exception("Impossible de préparer le job %s", job_id)
        return jsonify({"error": "Impossible de préparer le traitement."}), 500
    return jsonify({"job_id": job_id})


@app.route("/")
@app.route("/static/index.html")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/health")
def health():
    with jobs_lock:
        active = sum(1 for job in jobs.values() if job["status"] not in ("done", "error"))
    try:
        ytdlp_version = version("yt-dlp")
    except PackageNotFoundError:
        ytdlp_version = None
    return jsonify(status="ok", pid=os.getpid(), active_jobs=active + int(live_slot.locked()),
                   ytdlp_version=ytdlp_version)


@app.route("/languages")
def languages():
    return jsonify(SUPPORTED_LANGUAGES)


@app.route("/transcribe", methods=["POST"])
def transcribe():
    try:
        data = _json_object()
        options = _read_options(data)
        url = _read_url(data)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return _start_job(run_transcription, (url, options), slots=transcription_slots)


@app.route("/rediarize/<job_id>", methods=["POST"])
def rediarize(job_id):
    request.max_content_length = 8192
    try:
        speaker_options = _read_speaker_options(_json_object())
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    with jobs_lock:
        source_job = jobs.get(job_id)
        if source_job is None:
            return jsonify({"error": "Job introuvable ou expiré."}), 404
        if source_job["status"] != "done" or not source_job["rediarize_available"]:
            return jsonify({"error": "Un résultat horodaté avec son audio source est nécessaire."}), 409
        segments = source_job["segments"]
        job = _new_job(source_job["title"], kind="rediarization")
        job.update(speaker_options=speaker_options, duration_secs=source_job["duration_secs"],
                   language=source_job["language"], model_used=source_job["model_used"])
    return _start_job(run_rediarization, (segments, speaker_options), slots=transcription_slots,
                      job=job, source_audio=_job_directory(job_id) / "source.wav")


@app.route("/status/<job_id>")
def status(job_id):
    summary = request.args.get("summary") == "1"
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            return jsonify({"error": "Job introuvable ou expiré"}), 404
        snapshot = job.copy()
        logs = snapshot.pop("logs")
        if not summary:
            snapshot["logs"] = list(logs)
        snapshot.pop("log_count")
    if summary:
        for key in ("transcript", "segments", "speakers", "diarization_turns"):
            snapshot.pop(key)
    return jsonify(snapshot)


@app.route("/logs/<job_id>")
def get_logs(job_id):
    cursor = request.args.get("since", type=int)
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            return jsonify({"error": "Job introuvable ou expiré"}), 404
        logs, count = list(job["logs"]), job["log_count"]
    if cursor is not None:
        logs = logs[max(0, cursor - (count - len(logs))):]
    return jsonify({"logs": logs, "next": count})


@app.route("/test-diarize")
def test_diarize():
    if not gpu_lock.acquire(blocking=False):
        return jsonify({"ok": False, "error": "GPU occupé. Réessayez après le traitement en cours."})
    try:
        _get_diarization_pipeline()
        return jsonify({"ok": True, "message": "Pipeline pyannote chargé."})
    except Exception:
        app.logger.exception("Test pyannote impossible")
        return jsonify({"ok": False, "error": (
            "La détection des voix n’est pas disponible. Vérifiez l’accès à Community-1 "
            "sur Hugging Face et le jeton HF_TOKEN dans le fichier .env."
        )})
    finally:
        gpu_lock.release()


@app.route("/download-audio/<job_id>")
def download_audio(job_id):
    path = _audio_files.get(job_id)
    if not path or not os.path.isfile(path):
        return jsonify({"error": "Fichier audio non disponible"}), 404
    title = (jobs.get(job_id) or {}).get("title") or "audio"
    safe  = "".join(c if c.isalnum() or c in "- _" else "_" for c in title)
    return send_file(path, as_attachment=True, download_name=f"{safe}.mp3", mimetype="audio/mpeg")


@app.route("/download-video", methods=["POST"])
def download_video():
    try:
        data = _json_object()
        url = _read_url(data)
        quality = data.get("quality", "best")
        if not isinstance(quality, str) or quality not in VIDEO_QUALITIES:
            raise ValueError("Qualité non supportée.")
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    job = _new_job(kind="video")
    job.update(progress=None, video_available=False)
    return _start_job(run_video_download, (url, quality), slots=video_slots, job=job)


@app.route("/download-video-file/<job_id>")
def download_video_file(job_id):
    path = _video_files.get(job_id)
    if not path or not os.path.isfile(path):
        return jsonify({"error": "Fichier vidéo non disponible"}), 404
    title = (jobs.get(job_id) or {}).get("title") or "video"
    safe  = "".join(c if c.isalnum() or c in "- _" else "_" for c in title)
    ext   = os.path.splitext(path)[1] or ".mp4"
    return send_file(path, as_attachment=True, download_name=f"{safe}{ext}")


@app.route("/transcribe-file", methods=["POST"])
def transcribe_file():
    f = request.files.get("file")
    if f is None or not f.filename:
        return jsonify({"error": "Aucun fichier reçu."}), 400
    try:
        options = _read_options(request.form, form=True)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    title = Path(f.filename.replace("\\", "/")).stem[:200]
    return _start_job(
        run_transcription_from_file, (options,), slots=transcription_slots, job=_new_job(title), upload=f,
    )


@app.route("/speaker-audio/<job_id>/<speaker_id>")
def speaker_audio(job_id, speaker_id):
    clips = _speaker_clips.get(job_id, {})
    path  = clips.get(speaker_id)
    if not path or not os.path.isfile(path):
        return jsonify({"error": "Extrait audio non disponible"}), 404
    return send_file(path, mimetype="audio/mpeg")


@app.route("/api/transcribe-live", methods=["POST", "OPTIONS"])
def transcribe_live():
    if request.method == "OPTIONS":
        resp = app.make_default_options_response()
        resp.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return resp

    request.max_content_length = LIVE_UPLOAD_LIMIT
    if not live_slot.acquire(blocking=False):
        return jsonify({"error": "Une dictée est déjà en cours."}), 429
    try:
        f = request.files.get("audio")
        if f is None:
            return jsonify({"error": "Champ 'audio' manquant"}), 400
        try:
            language = _read_language(request.form)
            preprocess = _read_bool(request.form, "preprocess", False, form=True)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        JOB_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="lexia_live_", dir=JOB_ROOT) as tmpdir:
            src_path = os.path.join(tmpdir, "input")
            wav_path = os.path.join(tmpdir, "audio.wav")
            f.save(src_path)
            try:
                convert_to_wav(src_path, wav_path, LIVE_MAX_SECONDS + 1)
            except subprocess.CalledProcessError:
                return jsonify({"error": "Format audio non reconnu par FFmpeg."}), 422
            duration = sf.info(wav_path).duration
            if duration <= 0:
                return jsonify({"error": "L'audio est vide."}), 422
            if duration > LIVE_MAX_SECONDS:
                return jsonify({"error": "La dictée est limitée à 120 secondes."}), 413
            if preprocess:
                preprocess_audio(wav_path)

            if not gpu_lock.acquire(blocking=False):
                return jsonify({"error": "GPU occupé. Réessayez dans un instant."}), 503
            try:
                segments, info = _whisper_segments(wav_path, language)
                transcript = "\n".join(seg.text.strip() for seg in segments)
            finally:
                gpu_lock.release()
            return jsonify(transcript=transcript, language=info.language, duration_secs=round(duration, 2))
    except HTTPException:
        raise
    except subprocess.TimeoutExpired:
        return jsonify({"error": "La conversion audio a dépassé le délai maximal."}), 504
    except Exception:
        app.logger.exception("Erreur pendant la dictée")
        return jsonify({"error": "Erreur interne lors de la transcription."}), 500
    finally:
        live_slot.release()


if __name__ == "__main__":
    from waitress import serve

    ensure_port_free(SERVER_PORT)
    JOB_ROOT.mkdir(parents=True, exist_ok=True)
    if not shutil.which(FFMPEG_BIN):
        sys.exit("FFmpeg est introuvable dans le PATH.")
    print("Chargement de Whisper large-v3...")
    _whisper_model()
    threading.Thread(target=_cleanup_loop, daemon=True).start()
    serve(app, host="0.0.0.0", port=SERVER_PORT, threads=8,
          max_request_body_size=app.config["MAX_CONTENT_LENGTH"], connection_limit=50)
