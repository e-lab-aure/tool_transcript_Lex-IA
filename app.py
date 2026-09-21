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
_SCRIPTS   = os.path.dirname(sys.executable)
_YTDLP_EXE = os.path.join(_SCRIPTS, "yt-dlp.exe")
# Utilise le module pip du venv (cohérent avec pip install --upgrade yt-dlp)
YTDLP_BIN  = [sys.executable, "-m", "yt_dlp"]
FFMPEG_BIN = "ffmpeg"

YTDLP_UPDATE_TIMEOUT = 180  # secondes
# Extras officiels : "default" apporte yt-dlp-ejs (resolution des challenges JS
# YouTube), "deno" installe le runtime JS dans le venv, detecte automatiquement.
YTDLP_PACKAGE = "yt-dlp[default,deno]"


def update_ytdlp() -> None:
    """
    Met a jour yt-dlp, yt-dlp-ejs et deno via pip avant le chargement des modeles :
    les sites changent souvent et une version ancienne echoue (formats manquants, 403).
    Non bloquant : en cas d'echec (hors ligne...), la version installee est conservee.
    Desactivable avec LEXIA_SKIP_YTDLP_UPDATE=1 dans .env.
    """
    if os.environ.get("LEXIA_SKIP_YTDLP_UPDATE", "").strip() == "1":
        print("⏭  Mise à jour de yt-dlp ignorée (LEXIA_SKIP_YTDLP_UPDATE=1).")
        return

    print("⏳ Mise à jour de yt-dlp...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade", "--quiet",
             "--disable-pip-version-check", YTDLP_PACKAGE],
            capture_output=True, text=True, check=True, timeout=YTDLP_UPDATE_TIMEOUT,
        )
    except subprocess.CalledProcessError as e:
        detail = (e.stderr or e.stdout or "").strip().splitlines()
        print(f"⚠️  Mise à jour de yt-dlp échouée, version actuelle conservée : {detail[-1] if detail else e}")
    except subprocess.TimeoutExpired:
        print(f"⚠️  Mise à jour de yt-dlp interrompue après {YTDLP_UPDATE_TIMEOUT}s, version actuelle conservée.")

    try:
        version = subprocess.run(
            YTDLP_BIN + ["--version"], capture_output=True, text=True, timeout=30,
        ).stdout.strip()
        print(f"✅ yt-dlp {version or 'version inconnue'}")
    except (OSError, subprocess.TimeoutExpired) as e:
        print(f"⚠️  yt-dlp introuvable après mise à jour : {e}")

    if not shutil.which("deno", path=_SCRIPTS + os.pathsep + os.environ.get("PATH", "")):
        print("⚠️  deno introuvable : certains formats YouTube seront indisponibles (lancer update.bat).")


update_ytdlp()

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
jobs: dict          = {}
_audio_files: dict  = {}  # job_id -> chemin MP3 persistant
_speaker_clips: dict = {}  # job_id -> {speaker_id: chemin MP3 extrait}
_video_files: dict  = {}  # job_id -> chemin de la video telechargee

# Qualites proposees pour le telechargement video : cle -> hauteur max (None = meilleure)
VIDEO_QUALITIES = {"best": None, "1080": 1080, "720": 720, "480": 480}
VIDEO_DIR_PREFIX = "lexia_video_"


def _new_job(title: str | None = None, kind: str = "transcription") -> dict:
    return {
        "kind":       kind,
        "status":     "queued",
        "title":      title,
        "transcript": None,
        "language":   None,
        "model_used": None,
        "diarized":        False,
        "audio_available": False,
        "test_mode":       0,
        "speakers":        [],
        "segments":        [],
        "duration_secs":   None,
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

    # PyTorch >= 2.6 : weights_only=True par défaut bloque les classes pyannote.
    # On force weights_only=False uniquement pendant ce chargement (source HF officielle).
    _orig_load = torch.load
    torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, "weights_only": False})
    try:
        _diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
    finally:
        torch.load = _orig_load
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


def _build_speaker_meta(whisper_segments: list, diar_segments: list) -> list[dict]:
    """
    Construit les metadonnees par locuteur pour la modale de renommage.
    Retourne [{id, label, clip_start, clip_end, excerpt}, ...] dans l'ordre d'apparition.
    """
    seen: list = []
    for d in diar_segments:
        if d["speaker"] not in seen:
            seen.append(d["speaker"])

    meta = []
    for i, speaker_id in enumerate(seen):
        label = f"Locuteur {i + 1}"
        speaker_diar = [d for d in diar_segments if d["speaker"] == speaker_id]
        if not speaker_diar:
            continue

        # Premier segment de ce locuteur — clip de 8s maximum
        first = speaker_diar[0]
        clip_start = first["start"]
        clip_end   = min(first["end"], clip_start + 8.0)

        # Premier segment Whisper qui chevauche la region du clip
        excerpt = ""
        for seg in whisper_segments:
            if max(seg.start, clip_start - 1.0) < min(seg.end, clip_end + 5.0):
                excerpt = seg.text.strip()
                break

        meta.append({
            "id":         speaker_id,
            "label":      label,
            "clip_start": clip_start,
            "clip_end":   clip_end,
            "excerpt":    excerpt,
        })
    return meta


def _build_segments(whisper_segs: list, diar_segments: list | None = None) -> list[dict]:
    """
    Construit [{start, end, text, speaker}] a partir des segments Whisper.
    Si diar_segments fourni, ajoute le locuteur par max-overlap.
    """
    names: dict = {}
    if diar_segments:
        seen: list = []
        for d in diar_segments:
            if d["speaker"] not in seen:
                seen.append(d["speaker"])
        names = {sp: f"Locuteur {i + 1}" for i, sp in enumerate(seen)}

    result = []
    for seg in whisper_segs:
        speaker = None
        if diar_segments:
            best, best_ov = None, 0.0
            for d in diar_segments:
                ov = max(0.0, min(seg.end, d["end"]) - max(seg.start, d["start"]))
                if ov > best_ov:
                    best_ov, best = ov, d["speaker"]
            speaker = names.get(best, "Inconnu") if best else "Inconnu"
        result.append({
            "start":   round(seg.start, 3),
            "end":     round(seg.end,   3),
            "text":    seg.text.strip(),
            "speaker": speaker,
        })
    return result


def _save_speaker_clips(job_id: str, speaker_meta: list, wav_path: str) -> None:
    """Extrait un court extrait MP3 par locuteur depuis wav_path."""
    clips: dict = {}
    for speaker in speaker_meta:
        duration = speaker["clip_end"] - speaker["clip_start"]
        if duration <= 0:
            continue
        mp3_path = os.path.join(
            tempfile.gettempdir(),
            f"lexia_{job_id}_spk_{speaker['id']}.mp3",
        )
        try:
            subprocess.run(
                [FFMPEG_BIN, "-y",
                 "-ss", str(speaker["clip_start"]),
                 "-t",  str(duration),
                 "-i",  wav_path,
                 "-b:a", "64k", mp3_path],
                capture_output=True, check=True,
            )
            clips[speaker["id"]] = mp3_path
        except subprocess.CalledProcessError:
            pass
    _speaker_clips[job_id] = clips


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
        diar_segments = diarize_audio(wav_path, job_id)
        transcript = assign_speakers(segments, diar_segments)
        speaker_meta = _build_speaker_meta(segments, diar_segments)
        jobs[job_id]["speakers"]  = speaker_meta
        jobs[job_id]["segments"]  = _build_segments(segments, diar_segments)
        _save_speaker_clips(job_id, speaker_meta, wav_path)
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
        transcript, detected_lang, segs = transcribe_with_whisper(wav_path, None, job_id, return_segments=True)
        jobs[job_id]["segments"] = _build_segments(segs)
        return transcript, detected_lang, f"Whisper large-v3 ({detected_lang} détecté)"

    log(job_id, f"Langue explicite ({language}) → Whisper large-v3", "step")
    transcript, detected_lang, segs = transcribe_with_whisper(wav_path, language, job_id, return_segments=True)
    jobs[job_id]["segments"] = _build_segments(segs)
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
    test_mode: int = 0,
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
                _trim_wav(wav_path, test_mode)
                log(job_id, f"Mode test : audio tronqué à {test_mode} secondes.", "step")
                jobs[job_id]["test_mode"] = test_mode

            if preprocess:
                preprocess_audio(wav_path, job_id)

            try:
                _info = sf.info(wav_path)
                jobs[job_id]["duration_secs"] = round(_info.frames / _info.samplerate, 1)
            except Exception:
                pass

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
    test_mode: int = 0,
) -> None:
    jobs[job_id]["status"] = "downloading"
    log(job_id, "Démarrage du job de transcription", "step")
    log(job_id, f"URL cible : {url}", "info")
    log(job_id, f"Langue sélectionnée : {SUPPORTED_LANGUAGES.get(language, language)}", "info")

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            log(job_id, "Récupération du titre de la vidéo via yt-dlp...", "step")
            title_result = subprocess.run(
                YTDLP_BIN + ["--no-playlist", "--print", "title", url],
                capture_output=True, text=True,
            )
            
            title = title_result.stdout.strip().split("\n")[0] or "transcript"
            jobs[job_id]["title"] = title
            log(job_id, f"Titre : « {title} »", "info")

            log(job_id, "Téléchargement de la piste audio (yt-dlp → mp3)...", "step")
            dl = subprocess.run(
                YTDLP_BIN + ["-x", "--audio-format", "mp3", "--audio-quality", "0",
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
                _trim_wav(wav_path, test_mode)
                log(job_id, f"Mode test : audio tronqué à {test_mode} secondes.", "step")
                jobs[job_id]["test_mode"] = test_mode

            if preprocess:
                preprocess_audio(wav_path, job_id)

            try:
                _info = sf.info(wav_path)
                jobs[job_id]["duration_secs"] = round(_info.frames / _info.samplerate, 1)
            except Exception:
                pass

            _save_job_audio(job_id, wav_path)

            jobs[job_id]["status"] = "transcribing"
            transcript, detected_lang, model_used = _route_and_transcribe(
                wav_path, tmpdir, language, diarize, job_id
            )
            _finish_job(job_id, transcript, detected_lang, model_used)

    except Exception as e:
        _fail_job(job_id, f"Erreur : {e}\n{traceback.format_exc()}")


# ---------------------------------------------------------------------------
# Telechargement video (sans transcription)
# ---------------------------------------------------------------------------

# Ligne emise par yt-dlp via --progress-template, parsee pour la barre de progression
_PROGRESS_TAG = "LEXIA_PROGRESS"
_PROGRESS_TEMPLATE = (
    f"download:{_PROGRESS_TAG} %(progress.downloaded_bytes)s|%(progress.total_bytes)s"
    "|%(progress.total_bytes_estimate)s|%(progress.speed)s|%(progress.eta)s"
)


def _purge_stale_video_dirs() -> None:
    """Supprime les videos laissees par une execution precedente du serveur."""
    tmp_root = tempfile.gettempdir()
    for name in os.listdir(tmp_root):
        if name.startswith(VIDEO_DIR_PREFIX):
            shutil.rmtree(os.path.join(tmp_root, name), ignore_errors=True)


def _video_format_args(quality: str) -> list[str]:
    """
    Selection du format : meilleure video + meilleur audio fusionnes en MP4.
    Le tri -S privilegie la resolution, puis a resolution egale le h264 et les
    conteneurs MP4/M4A : fichier lisible partout, sans re-encodage a la fusion.
    Sur YouTube, cette selection requiert deno (sinon seuls des formats degrades
    et souvent refuses en 403 sont proposes).
    """
    height = VIDEO_QUALITIES[quality]
    res  = f"res:{height}" if height else "res"
    sort = f"{res},vcodec:h264,ext:mp4:m4a"
    return ["-f", "bv*+ba/b", "-S", sort, "--merge-output-format", "mp4"]


def _parse_progress(line: str) -> dict | None:
    """Convertit une ligne LEXIA_PROGRESS en {percent, speed, eta} (valeurs NA -> None)."""
    fields = line.split(_PROGRESS_TAG, 1)[1].strip().split("|")
    if len(fields) != 5:
        return None

    def num(value: str) -> float | None:
        try:
            return float(value)
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


def run_video_download(job_id: str, url: str, quality: str) -> None:
    job = jobs[job_id]
    job["status"] = "downloading"
    log(job_id, "Démarrage du téléchargement vidéo", "step")
    log(job_id, f"URL cible : {url}", "info")
    log(job_id, f"Qualité demandée : {'meilleure disponible' if quality == 'best' else quality + 'p max'}", "info")

    out_dir = tempfile.mkdtemp(prefix=VIDEO_DIR_PREFIX)
    # Force l'UTF-8 sur la sortie de yt-dlp (titres accentues sous Windows)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    try:
        log(job_id, "Récupération du titre de la vidéo via yt-dlp...", "step")
        title_result = subprocess.run(
            YTDLP_BIN + ["--no-playlist", "--print", "title", "--", url],
            capture_output=True, text=True, encoding="utf-8", errors="replace", env=env,
        )
        if title_result.returncode != 0:
            _fail_job(job_id, title_result.stderr.strip() or "URL non prise en charge par yt-dlp")
            shutil.rmtree(out_dir, ignore_errors=True)
            return
        job["title"] = title_result.stdout.strip().split("\n")[0] or "video"
        log(job_id, f"Titre : « {job['title']} »", "info")

        log(job_id, "Téléchargement des flux vidéo et audio...", "step")
        cmd = YTDLP_BIN + _video_format_args(quality) + [
            "--no-playlist", "--newline", "--no-colors",
            "--progress-template", _PROGRESS_TEMPLATE,
            "-o", os.path.join(out_dir, "video.%(ext)s"),
            "--", url,
        ]
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", env=env,
        )
        errors = []
        stream = 0
        for raw in proc.stdout:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(_PROGRESS_TAG):
                progress = _parse_progress(line)
                if progress:
                    job["progress"] = {**progress, "stream": stream}
            elif line.startswith("[download] Destination:"):
                # Un flux par destination : video puis audio quand ils sont separes
                stream += 1
                log(job_id, f"Flux {stream} : {os.path.basename(line.split(':', 1)[1].strip())}", "model")
            elif line.startswith("[Merger]"):
                job["status"] = "merging"
                log(job_id, "Fusion vidéo + audio en MP4 (ffmpeg)...", "step")
            elif line.startswith("ERROR:"):
                errors.append(line)
                log(job_id, line, "error")
            elif line.startswith("WARNING:"):
                log(job_id, line, "info")
        proc.wait()

        if proc.returncode != 0:
            _fail_job(job_id, "\n".join(errors) or "Erreur lors du téléchargement")
            shutil.rmtree(out_dir, ignore_errors=True)
            return

        # Seul le fichier final subsiste apres la fusion ; on ignore les residus partiels
        candidates = [
            os.path.join(out_dir, f) for f in os.listdir(out_dir)
            if not f.endswith((".part", ".ytdl", ".temp"))
        ]
        if not candidates:
            _fail_job(job_id, "Fichier vidéo introuvable après téléchargement")
            shutil.rmtree(out_dir, ignore_errors=True)
            return
        video_path = max(candidates, key=os.path.getsize)

        _video_files[job_id] = video_path
        size_mb = os.path.getsize(video_path) / 1024 / 1024
        job.update({
            "status":          "done",
            "video_available": True,
            "video_ext":       os.path.splitext(video_path)[1].lstrip(".") or "mp4",
            "file_size_mb":    round(size_mb, 1),
            "progress":        {"percent": 100.0, "speed": None, "eta": 0, "stream": stream},
        })
        log(job_id, f"Vidéo prête : {os.path.basename(video_path)} ({size_mb:.1f} Mo)", "success")

    except Exception as e:
        _fail_job(job_id, f"Erreur : {e}\n{traceback.format_exc()}")
        shutil.rmtree(out_dir, ignore_errors=True)


_purge_stale_video_dirs()


# ---------------------------------------------------------------------------
# Routes Flask
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/health")
def health():
    """Sonde utilisee par le lanceur : le serveur ne repond qu'apres le chargement
    des modeles, et signale les transcriptions en cours avant un arret."""
    active = sum(1 for job in jobs.values() if job["status"] not in ("done", "error"))
    return jsonify({"status": "ok", "active_jobs": active})


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
    test_mode  = int(data.get("test_mode", 0))

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


@app.route("/download-video", methods=["POST"])
def download_video():
    data    = request.get_json(silent=True) or {}
    url     = str(data.get("url", "")).strip()
    quality = str(data.get("quality", "best")).strip()

    if not url:
        return jsonify({"error": "URL manquante"}), 400
    # Refuse tout ce qui n'est pas une URL web (evite qu'une valeur soit lue comme option yt-dlp)
    if not url.lower().startswith(("http://", "https://")):
        return jsonify({"error": "URL invalide : http:// ou https:// attendu"}), 400
    if quality not in VIDEO_QUALITIES:
        return jsonify({"error": f"Qualité non supportée : {quality}"}), 400

    job_id = str(uuid.uuid4())
    jobs[job_id] = _new_job(kind="video")
    jobs[job_id].update({"progress": None, "video_available": False})
    threading.Thread(
        target=run_video_download,
        args=(job_id, url, quality),
        daemon=True,
    ).start()
    return jsonify({"job_id": job_id})


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
    if "file" not in request.files:
        return jsonify({"error": "Aucun fichier reçu"}), 400

    f          = request.files["file"]
    language   = request.form.get("language", "auto").strip()
    preprocess = request.form.get("preprocess", "true").lower() == "true"
    diarize    = request.form.get("diarize", "false").lower() == "true"
    test_mode  = int(request.form.get("test_mode", "0") or "0")

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


@app.route("/speaker-audio/<job_id>/<speaker_id>")
def speaker_audio(job_id, speaker_id):
    clips = _speaker_clips.get(job_id, {})
    path  = clips.get(speaker_id)
    if not path or not os.path.isfile(path):
        return jsonify({"error": "Extrait audio non disponible"}), 404
    return send_file(path, mimetype="audio/mpeg")


@app.route("/api/transcribe-live", methods=["POST", "OPTIONS"])
def transcribe_live():
    # Pre-flight CORS
    if request.method == "OPTIONS":
        resp = app.make_default_options_response()
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return resp

    if "audio" not in request.files:
        return jsonify({"error": "Champ 'audio' manquant"}), 400

    f          = request.files["audio"]
    language   = request.form.get("language", "auto").strip()
    preprocess = request.form.get("preprocess", "false").lower() == "true"

    if language not in SUPPORTED_LANGUAGES:
        return jsonify({"error": f"Langue non supportee : {language}"}), 400

    tmpdir = tempfile.mkdtemp(prefix="live_")
    try:
        ext      = os.path.splitext(f.filename or "audio.webm")[1].lower() or ".webm"
        src_path = os.path.join(tmpdir, f"input{ext}")
        wav_path = os.path.join(tmpdir, "audio.wav")
        f.save(src_path)

        try:
            convert_to_wav(src_path, wav_path)
        except subprocess.CalledProcessError:
            return jsonify({"error": "Format audio non reconnu par ffmpeg"}), 422

        audio_data, sr = sf.read(wav_path, dtype="float32")
        duration_secs  = len(audio_data) / sr

        if preprocess:
            preprocess_audio(wav_path)

        lang_param = None if language == "auto" else language
        transcript, detected_lang = transcribe_with_whisper(wav_path, lang_param)

        resp = jsonify({
            "transcript":   transcript,
            "language":     detected_lang,
            "duration_secs": round(duration_secs, 2),
        })
        resp.headers["Access-Control-Allow-Origin"] = "*"
        return resp

    except Exception as e:
        print(f"[live] Erreur : {e}")
        return jsonify({"error": "Erreur interne lors de la transcription"}), 500

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=False, port=5000, threaded=True)
