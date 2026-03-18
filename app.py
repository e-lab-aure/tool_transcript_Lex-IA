import os
import sys
import tempfile
import subprocess
import threading
import uuid
import math
import numpy as np
import soundfile as sf
from flask import Flask, request, jsonify, send_from_directory
from datetime import datetime

app = Flask(__name__, static_folder="static")

# ---------------------------------------------------------------------------
# Chemins des executables
# ---------------------------------------------------------------------------
_SCRIPTS = os.path.dirname(sys.executable)
YTDLP_BIN = os.path.join(_SCRIPTS, "yt-dlp.exe")
if not os.path.isfile(YTDLP_BIN):
    YTDLP_BIN = "yt-dlp"
FFMPEG_BIN = "ffmpeg"

# ---------------------------------------------------------------------------
# Chargement des modeles au demarrage
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
# Parametres
# ---------------------------------------------------------------------------
CANARY_CHUNK_SECS = 40
SAMPLE_RATE       = 16000

jobs = {}

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
# Logging par job
# ---------------------------------------------------------------------------

def log(job_id: str, msg: str, level: str = "info"):
    ts = datetime.now().strftime("%H:%M:%S")
    entry = {"ts": ts, "msg": msg, "level": level}
    jobs[job_id]["logs"].append(entry)
    print(f"[{ts}] [{level.upper()}] {msg}")


# ---------------------------------------------------------------------------
# Helpers audio
# ---------------------------------------------------------------------------

def convert_to_wav(src_path: str, dst_path: str):
    subprocess.run(
        [FFMPEG_BIN, "-y", "-i", src_path,
         "-ar", str(SAMPLE_RATE), "-ac", "1", dst_path],
        capture_output=True,
        check=True,
    )


def split_audio(wav_path: str, chunk_secs: int, tmpdir: str):
    audio, sr = sf.read(wav_path, dtype="float32")
    assert sr == SAMPLE_RATE
    total_secs = len(audio) / sr
    chunk_samples = chunk_secs * sr
    n_chunks = math.ceil(len(audio) / chunk_samples)
    chunk_paths = []
    for i in range(n_chunks):
        start = i * chunk_samples
        end   = min((i + 1) * chunk_samples, len(audio))
        chunk = audio[start:end]
        chunk_path = os.path.join(tmpdir, f"chunk_{i:04d}.wav")
        sf.write(chunk_path, chunk, sr)
        chunk_paths.append(chunk_path)
    return chunk_paths, total_secs


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
            prompts=[
                [{
                    "role": "user",
                    "content": f"Transcribe the following: {canary_model.audio_locator_tag}",
                    "audio": [chunk_path],
                }]
            ],
            max_new_tokens=1024,
        )
        text = canary_model.tokenizer.ids_to_text(answer_ids[0].cpu()).strip()
        log(job_id, f"Chunk {i+1}/{n} transcrit ({len(text)} caractères)", "success")
        parts.append(text)

    log(job_id, f"Assemblage des {n} chunks en transcript final...", "step")
    return " ".join(parts)


def transcribe_with_whisper(wav_path: str, language, job_id: str = None):
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
    return transcript, info.language


# ---------------------------------------------------------------------------
# Job de transcription
# ---------------------------------------------------------------------------

def run_transcription_from_file(job_id: str, file_path: str, upload_dir: str, language: str):
    """Même pipeline que run_transcription mais à partir d'un fichier local uploadé."""
    import shutil
    jobs[job_id]["status"] = "transcribing"
    title = jobs[job_id]["title"]
    log(job_id, "Démarrage du job depuis un fichier local", "step")
    log(job_id, f"Fichier : {os.path.basename(file_path)}", "info")
    log(job_id, f"Langue sélectionnée : {SUPPORTED_LANGUAGES.get(language, language)}", "info")

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            size_mb = os.path.getsize(file_path) / 1024 / 1024
            log(job_id, f"Taille du fichier : {size_mb:.1f} Mo", "info")

            # Conversion WAV
            log(job_id, "Conversion en WAV 16kHz mono via ffmpeg...", "step")
            wav_path = os.path.join(tmpdir, "audio.wav")
            try:
                convert_to_wav(file_path, wav_path)
                wav_mb = os.path.getsize(wav_path) / 1024 / 1024
                log(job_id, f"WAV généré : {wav_mb:.1f} Mo @ {SAMPLE_RATE}Hz mono", "success")
            except subprocess.CalledProcessError as e:
                log(job_id, f"Erreur ffmpeg : {e.stderr.decode()}", "error")
                jobs[job_id]["status"] = "error"
                jobs[job_id]["error"] = f"Erreur ffmpeg : {e.stderr.decode()}"
                return

            jobs[job_id]["status"] = "transcribing"

            if language == "en":
                log(job_id, "Langue : anglais forcé → routage vers Canary-Qwen-2.5B", "step")
                transcript = transcribe_with_canary(wav_path, tmpdir, job_id)
                detected_lang = "en"
                model_used = "Canary-Qwen-2.5B"

            elif language == "auto":
                log(job_id, "Détection automatique de la langue via Whisper large-v3...", "step")
                jobs[job_id]["status"] = "detecting"
                _, detected_lang = transcribe_with_whisper(wav_path, None)
                log(job_id, f"Langue détectée : {detected_lang}", "info")

                if detected_lang == "en":
                    log(job_id, "Anglais détecté → routage vers Canary-Qwen-2.5B", "step")
                    jobs[job_id]["status"] = "transcribing"
                    transcript = transcribe_with_canary(wav_path, tmpdir, job_id)
                    model_used = "Canary-Qwen-2.5B (anglais détecté)"
                else:
                    log(job_id, f"Langue non-anglaise ({detected_lang}) → Whisper large-v3", "step")
                    jobs[job_id]["status"] = "transcribing"
                    transcript, detected_lang = transcribe_with_whisper(wav_path, None, job_id)
                    model_used = f"Whisper large-v3 ({detected_lang} détecté)"
            else:
                log(job_id, f"Langue explicite ({language}) → Whisper large-v3", "step")
                transcript, detected_lang = transcribe_with_whisper(wav_path, language, job_id)
                model_used = "Whisper large-v3"

            word_count = len(transcript.split())
            log(job_id, f"Transcription terminée : {word_count} mots", "success")
            log(job_id, f"Modèle utilisé : {model_used}", "info")

            jobs[job_id]["status"] = "done"
            jobs[job_id]["transcript"] = transcript
            jobs[job_id]["language"] = detected_lang
            jobs[job_id]["model_used"] = model_used

    except Exception as e:
        import traceback
        log(job_id, f"Erreur : {e}", "error")
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)
    finally:
        # Nettoyage du dossier d'upload temporaire
        shutil.rmtree(upload_dir, ignore_errors=True)


def run_transcription(job_id: str, url: str, language: str):
    jobs[job_id]["status"] = "downloading"
    log(job_id, "Démarrage du job de transcription", "step")
    log(job_id, f"URL cible : {url}", "info")
    log(job_id, f"Langue sélectionnée : {SUPPORTED_LANGUAGES.get(language, language)}", "info")

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_template = os.path.join(tmpdir, "audio.%(ext)s")

        # Titre
        log(job_id, "Récupération du titre de la vidéo via yt-dlp...", "step")
        title_result = subprocess.run(
            [YTDLP_BIN, "--no-playlist", "--print", "title", url],
            capture_output=True, text=True,
        )
        title = title_result.stdout.strip().split("\n")[0] if title_result.stdout.strip() else "transcript"
        jobs[job_id]["title"] = title
        log(job_id, f"Titre : « {title} »", "info")

        # Téléchargement audio
        log(job_id, "Téléchargement de la piste audio (yt-dlp → mp3)...", "step")
        result = subprocess.run(
            [YTDLP_BIN, "-x", "--audio-format", "mp3", "--audio-quality", "0",
             "-o", audio_template, "--no-playlist", url],
            capture_output=True, text=True,
        )

        if result.returncode != 0:
            log(job_id, f"Erreur yt-dlp : {result.stderr.strip()}", "error")
            jobs[job_id]["status"] = "error"
            jobs[job_id]["error"] = result.stderr.strip() or "Erreur lors du téléchargement"
            return

        audio_file = next(
            (os.path.join(tmpdir, f) for f in os.listdir(tmpdir)
             if f.endswith((".mp3", ".m4a", ".webm", ".opus"))),
            None
        )
        if not audio_file:
            all_files = os.listdir(tmpdir)
            log(job_id, f"Fichier audio introuvable. Présents : {all_files}", "error")
            jobs[job_id]["status"] = "error"
            jobs[job_id]["error"] = f"Fichier audio introuvable. Fichiers présents : {all_files}"
            return

        size_mb = os.path.getsize(audio_file) / 1024 / 1024
        log(job_id, f"Audio téléchargé : {os.path.basename(audio_file)} ({size_mb:.1f} Mo)", "success")

        # Conversion WAV
        log(job_id, f"Conversion en WAV 16kHz mono (requis par les modèles ASR)...", "step")
        wav_path = os.path.join(tmpdir, "audio.wav")
        try:
            convert_to_wav(audio_file, wav_path)
            wav_mb = os.path.getsize(wav_path) / 1024 / 1024
            log(job_id, f"WAV généré : {wav_mb:.1f} Mo @ {SAMPLE_RATE}Hz mono", "success")
        except subprocess.CalledProcessError as e:
            log(job_id, f"Erreur ffmpeg : {e.stderr.decode()}", "error")
            jobs[job_id]["status"] = "error"
            jobs[job_id]["error"] = f"Erreur ffmpeg : {e.stderr.decode()}"
            return

        jobs[job_id]["status"] = "transcribing"

        try:
            if language == "en":
                log(job_id, "Langue : anglais forcé → routage vers Canary-Qwen-2.5B", "step")
                transcript = transcribe_with_canary(wav_path, tmpdir, job_id)
                detected_lang = "en"
                model_used = "Canary-Qwen-2.5B"

            elif language == "auto":
                log(job_id, "Détection automatique de la langue via Whisper large-v3...", "step")
                jobs[job_id]["status"] = "detecting"
                _, detected_lang = transcribe_with_whisper(wav_path, None)
                log(job_id, f"Langue détectée : {detected_lang}", "info")

                if detected_lang == "en":
                    log(job_id, "Anglais détecté → routage vers Canary-Qwen-2.5B (meilleure qualité)", "step")
                    jobs[job_id]["status"] = "transcribing"
                    transcript = transcribe_with_canary(wav_path, tmpdir, job_id)
                    model_used = "Canary-Qwen-2.5B (anglais détecté)"
                else:
                    log(job_id, f"Langue non-anglaise ({detected_lang}) → routage vers Whisper large-v3", "step")
                    jobs[job_id]["status"] = "transcribing"
                    transcript, detected_lang = transcribe_with_whisper(wav_path, None, job_id)
                    model_used = f"Whisper large-v3 ({detected_lang} détecté)"

            else:
                log(job_id, f"Langue explicite ({language}) → Whisper large-v3", "step")
                transcript, detected_lang = transcribe_with_whisper(wav_path, language, job_id)
                model_used = "Whisper large-v3"

            word_count = len(transcript.split())
            log(job_id, f"Transcription terminée : {word_count} mots", "success")
            log(job_id, f"Modèle utilisé : {model_used}", "info")

            jobs[job_id]["status"] = "done"
            jobs[job_id]["transcript"] = transcript
            jobs[job_id]["language"] = detected_lang
            jobs[job_id]["model_used"] = model_used

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            log(job_id, f"Erreur : {e}", "error")
            log(job_id, tb, "error")
            jobs[job_id]["status"] = "error"
            jobs[job_id]["error"] = str(e)


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
    data = request.get_json() or {}
    url = data.get("url", "").strip()
    language = data.get("language", "auto").strip()

    if not url:
        return jsonify({"error": "URL manquante"}), 400
    if language not in SUPPORTED_LANGUAGES:
        return jsonify({"error": f"Langue non supportée : {language}"}), 400

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "queued",
        "title": None,
        "transcript": None,
        "language": None,
        "model_used": None,
        "error": None,
        "logs": [],
    }

    thread = threading.Thread(
        target=run_transcription,
        args=(job_id, url, language),
        daemon=True,
    )
    thread.start()

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



@app.route("/transcribe-file", methods=["POST"])
def transcribe_file():
    if 'file' not in request.files:
        return jsonify({"error": "Aucun fichier reçu"}), 400

    f = request.files['file']
    language = request.form.get('language', 'auto').strip()

    if not f.filename:
        return jsonify({"error": "Nom de fichier invalide"}), 400
    if language not in SUPPORTED_LANGUAGES:
        return jsonify({"error": f"Langue non supportée : {language}"}), 400

    # Sauvegarder le fichier dans un dossier temporaire persistant le temps du job
    upload_dir = tempfile.mkdtemp(prefix="upload_")
    ext = os.path.splitext(f.filename)[1].lower() or '.bin'
    safe_path = os.path.join(upload_dir, f"input{ext}")
    f.save(safe_path)

    title = os.path.splitext(f.filename)[0]  # nom du fichier sans extension = titre

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "queued",
        "title": title,
        "transcript": None,
        "language": None,
        "model_used": None,
        "error": None,
        "logs": [],
    }

    thread = threading.Thread(
        target=run_transcription_from_file,
        args=(job_id, safe_path, upload_dir, language),
        daemon=True,
    )
    thread.start()

    return jsonify({"job_id": job_id})

if __name__ == "__main__":
    app.run(debug=False, port=5000, threaded=True)
