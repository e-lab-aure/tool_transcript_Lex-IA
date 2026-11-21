import io
import subprocess
import tempfile
import threading
import time
import unittest
import wave
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import app


def sample_wav():
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", ""))
        wav.writeframes(b"\0\0" * 16000)
    return output.getvalue()


class AppTests(unittest.TestCase):
    def setUp(self):
        self.client = app.app.test_client()

    def test_health_reports_installed_ytdlp_version(self):
        health = self.client.get("/health")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json["ytdlp_version"], version("yt-dlp"))

        with patch.object(app, "version", side_effect=PackageNotFoundError("yt-dlp")):
            health = self.client.get("/health")
        self.assertEqual(health.json["status"], "ok")
        self.assertIsNone(health.json["ytdlp_version"])

    def test_startup_ytdlp_update_uses_local_python_and_handles_timeout(self):
        with (patch.dict(app.os.environ, {"LEXIA_SKIP_YTDLP_UPDATE": "0"}),
              patch.object(app.sys, "base_prefix", "base-python"),
              patch.object(app.subprocess, "run", side_effect=subprocess.TimeoutExpired("pip", 60)) as run,
              patch.object(app.app.logger, "warning") as warning):
            app._update_ytdlp()

        command = run.call_args.args[0]
        self.assertEqual(command[:4], [app.sys.executable, "-m", "pip", "install"])
        self.assertIn("--upgrade", command)
        self.assertIn("yt-dlp[default,deno]", command)
        self.assertEqual(run.call_args.kwargs["timeout"], 60)
        warning.assert_called_once()
        self.assertEqual(self.client.get("/health").json["ytdlp_version"], version("yt-dlp"))

    def test_startup_ytdlp_update_respects_opt_out(self):
        with (patch.dict(app.os.environ, {"LEXIA_SKIP_YTDLP_UPDATE": "1"}),
              patch.object(app.subprocess, "run") as run):
            app._update_ytdlp()
        run.assert_not_called()

    def test_invalid_options_return_json_400(self):
        for payload in ([], {"url": None}, {"url": "https://example.org", "test_mode": "abc"},
                        {"url": "https://example.org", "diarize": "false"},
                        {"url": "https://example.org", "diarize": True, "min_speakers": 3,
                         "max_speakers": 2}):
            with self.subTest(payload=payload):
                response = self.client.post("/transcribe", json=payload)
                self.assertEqual(response.status_code, 400)
                self.assertIn("error", response.json)
        response = self.client.post("/download-video", json=["invalide"])
        self.assertEqual(response.status_code, 400)

    def test_speaker_options_reach_url_job(self):
        with patch.object(app, "_start_job", return_value=("", 200)) as submit:
            response = self.client.post("/transcribe", json={
                "url": "https://example.org/audio", "diarize": True, "max_speakers": 3,
            })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(submit.call_args.args[1][1]["speaker_options"], {"max_speakers": 3})

    def test_audio_job_passes_speaker_count_and_expires(self):
        info = SimpleNamespace(language="fr")
        seg = SimpleNamespace(text="Bonjour.", start=0.0, end=0.7, words=None)
        with tempfile.TemporaryDirectory() as root:
            with (patch.object(app, "JOB_ROOT", Path(root)),
                  patch.object(app, "_whisper_segments", return_value=(iter([seg]), info)),
                  patch.object(app, "diarize_audio", return_value=[
                      {"start": 0.0, "end": 0.7, "speaker": "SPEAKER_00"}
                  ]) as diarize):
                response = self.client.post("/transcribe-file", data={
                    "file": (io.BytesIO(sample_wav()), "essai.wav"), "language": "fr",
                    "preprocess": "false", "diarize": "true", "num_speakers": "1",
                })
                self.assertEqual(response.status_code, 200)
                job_id = response.json["job_id"]
                deadline = time.monotonic() + 10
                while job_id not in app.finished_at and time.monotonic() < deadline:
                    time.sleep(0.02)
                status = self.client.get(f"/status/{job_id}").json
                self.assertEqual(status["status"], "done", status)
                self.assertEqual(status["segments"][0]["speaker_id"], "SPEAKER_00")
                self.assertEqual(status["speakers"][0]["excerpt"], "Bonjour.")
                self.assertEqual(diarize.call_args.args[2], {"num_speakers": 1})
                self.assertNotIn("transcript", self.client.get(f"/status/{job_id}?summary=1").json)
                logs = self.client.get(f"/logs/{job_id}?since=0").json
                self.assertTrue(logs["logs"])
                self.assertEqual(self.client.get(f"/logs/{job_id}?since={logs['next']}").json["logs"], [])
                with self.client.get(f"/download-audio/{job_id}") as audio:
                    self.assertEqual(audio.status_code, 200)
                with self.client.get(f"/speaker-audio/{job_id}/SPEAKER_00") as clip:
                    self.assertEqual(clip.status_code, 200)
                with patch.object(app, "JOB_TTL", 0):
                    app._purge_jobs()
                self.assertFalse((Path(root) / job_id).exists())
                self.assertEqual(self.client.get(f"/status/{job_id}").status_code, 404)

    def test_voice_detection_failure_preserves_transcript_and_allows_retry(self):
        segment = SimpleNamespace(text="Bonjour.", start=0.0, end=0.7, words=None)
        with tempfile.TemporaryDirectory() as root:
            with (patch.object(app, "JOB_ROOT", Path(root)),
                  patch.object(app, "_whisper_segments", return_value=(
                      iter([segment]), SimpleNamespace(language="fr")
                  )) as whisper,
                  patch.object(app, "diarize_audio", side_effect=RuntimeError("Modèle indisponible")) as diarize,
                  patch.object(app.app.logger, "exception")):
                response = self.client.post("/transcribe-file", data={
                    "file": (io.BytesIO(sample_wav()), "essai.wav"), "language": "fr",
                    "preprocess": "false", "diarize": "true",
                })
                self.assertEqual(response.status_code, 200)
                job_id = response.json["job_id"]
                path = f"/status/{job_id}"
                app.transcription_queue.join()
                original = self.client.get(path).json
                self.assertEqual(original["status"], "done")
                self.assertEqual(original["transcript"], "Bonjour.")
                self.assertEqual(original["segments"][0]["end"], 0.7)
                self.assertFalse(original["diarized"])
                self.assertTrue(original["rediarize_available"])
                self.assertIn("relancez-la", original["diarization_warning"])

                response = self.client.post(f"/rediarize/{job_id}", json={"num_speakers": 1})
                self.assertEqual(response.status_code, 200)
                app.transcription_queue.join()
                retry = self.client.get(f"/status/{response.json['job_id']}").json
                self.assertEqual(retry["status"], "error")
                self.assertIn("réglages avancés", retry["error"])
                self.assertEqual(self.client.get(path).json, original)
                whisper.assert_called_once()
                self.assertEqual(diarize.call_count, 2)
                with patch.object(app, "JOB_TTL", 0):
                    app._purge_jobs()

    def test_automatic_routing_does_not_decode_twice(self):
        with patch.object(app, "jobs", {"job": app._new_job()}):
            seg = SimpleNamespace(text="Bonjour.", start=0, end=1)
            fr = SimpleNamespace(language="fr", duration_after_vad=1)
            with patch.object(app, "_whisper_segments", return_value=(iter([seg]), fr)) as whisper:
                result = app._route_and_transcribe("test.wav", "", "", "auto", False, {}, "job")
            self.assertEqual(result[:2], ("Bonjour.", "fr"))
            whisper.assert_called_once()

            def unused_segments():
                self.fail("Whisper decoded English before Canary")
                yield

            en = SimpleNamespace(language="en", duration_after_vad=1)
            with (patch.object(app, "_whisper_segments", return_value=(unused_segments(), en)),
                  patch.object(app, "_canary_model", return_value=object()),
                  patch.object(app, "transcribe_with_canary", return_value="Hello.")):
                result = app._route_and_transcribe("test.wav", "", "", "auto", False, {}, "job")
            self.assertEqual(result[:2], ("Hello.", "en"))

    def test_timed_english_uses_whisper_even_without_diarization(self):
        segment = SimpleNamespace(text="Hello.", start=0, end=1)
        with (patch.object(app, "jobs", {"job": app._new_job()}),
              patch.object(app, "_whisper_segments", return_value=(
                  iter([segment]), SimpleNamespace(language="en")
              )),
              patch.object(app, "_canary_model") as canary):
            result = app._route_and_transcribe("audio.wav", "", "", "en", False, {}, "job", True)
            self.assertEqual(result, ("Hello.", "en", "Whisper large-v3"))
            self.assertEqual(app.jobs["job"]["segments"][0]["end"], 1)
            canary.assert_not_called()

    def test_rediarization_reuses_source_after_original_job_expires(self):
        seg = SimpleNamespace(text="Bonjour.", start=0.0, end=0.7)
        with tempfile.TemporaryDirectory() as root:
            with (patch.object(app, "JOB_ROOT", Path(root)),
                  patch.object(app, "_whisper_segments", return_value=(
                      iter([seg]), SimpleNamespace(language="fr")
                  )) as whisper,
                  patch.object(app, "diarize_audio") as diarize):
                def recognize(path, job_id, options):
                    self.assertTrue(path.is_file())
                    self.assertEqual(options, {"max_speakers": 1})
                    return [{"start": 0, "end": .7, "speaker": "NOUVELLE_VOIX"}]

                diarize.side_effect = recognize
                response = self.client.post("/transcribe-file", data={
                    "file": (io.BytesIO(sample_wav()), "essai.wav"), "language": "fr",
                    "preprocess": "false",
                })
                source_id = response.json["job_id"]
                app.transcription_queue.join()
                source = self.client.get(f"/status/{source_id}").json
                self.assertTrue(source["rediarize_available"])
                self.assertEqual(self.client.post(f"/rediarize/{source_id}", json={"max_speakers": 0}).status_code, 400)
                with app.gpu_lock:
                    response = self.client.post(f"/rediarize/{source_id}", json={"max_speakers": 1})
                    self.assertEqual(response.status_code, 200)
                    variant_id = response.json["job_id"]
                    app.finished_at[source_id] = 0
                    app._purge_jobs()
                    self.assertEqual(self.client.get(f"/status/{source_id}").status_code, 404)
                app.transcription_queue.join()
                variant = self.client.get(f"/status/{variant_id}").json
                self.assertEqual(variant["status"], "done")
                self.assertEqual(variant["segments"][0]["speaker_id"], "NOUVELLE_VOIX")
                self.assertEqual(variant["diarization_turns"][0]["speaker"], "NOUVELLE_VOIX")
                self.assertEqual(variant["segments"][0]["text"], "Bonjour.")
                self.assertEqual(variant["segments"][0]["end"], source["segments"][0]["end"])
                whisper.assert_called_once()
                diarize.assert_called_once()
                with self.client.get(f"/speaker-audio/{variant_id}/NOUVELLE_VOIX") as clip:
                    self.assertEqual(clip.status_code, 200)
                with patch.object(app, "JOB_TTL", 0):
                    app._purge_jobs()

    def test_words_on_different_sides_of_turn_are_assigned_separately(self):
        words = [SimpleNamespace(start=0, end=1, word=" Bonjour"),
                 SimpleNamespace(start=1, end=2, word=" Salut")]
        turns = [{"start": 0, "end": 1, "speaker": "A"}, {"start": 1, "end": 2, "speaker": "B"}]
        segments = app._build_segments([SimpleNamespace(words=words)], turns)
        self.assertEqual([(s["speaker_id"], s["text"]) for s in segments],
                         [("A", "Bonjour"), ("B", "Salut")])

    def test_transcription_queue_is_bounded_and_preserves_order(self):
        release = threading.Event()
        started = []

        def run(job_id, url, options):
            started.append(url)
            release.wait(5)
            app.jobs[job_id]["status"] = "done"

        with tempfile.TemporaryDirectory() as root:
            with (patch.object(app, "JOB_ROOT", Path(root)),
                  patch.object(app, "jobs", {}), patch.object(app, "finished_at", {}),
                  patch.object(app, "run_transcription", side_effect=run)):
                urls = [f"https://example.org/{index}" for index in range(5)]
                try:
                    responses = [self.client.post("/transcribe", json={"url": url}) for url in urls]
                    self.assertEqual([r.status_code for r in responses], [200, 200, 200, 200, 429])
                    self.assertLessEqual(len(started), 1)
                finally:
                    release.set()
                    app.transcription_queue.join()
                self.assertEqual(started, urls[:4])

    def test_live_busy_and_upload_limit_have_cors(self):
        with app.gpu_lock, tempfile.TemporaryDirectory() as root:
            with patch.object(app, "JOB_ROOT", Path(root)):
                response = self.client.post("/api/transcribe-live", data={
                    "audio": (io.BytesIO(sample_wav()), "essai.wav"),
                })
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.headers.get("Access-Control-Allow-Origin"), "*")
        with patch.object(app, "LIVE_UPLOAD_LIMIT", 2048):
            response = self.client.post("/api/transcribe-live", data={
                "audio": (io.BytesIO(sample_wav()), "essai.wav"),
            })
        self.assertEqual(response.status_code, 413)
        self.assertTrue(response.is_json)
        self.assertEqual(response.headers.get("Access-Control-Allow-Origin"), "*")


if __name__ == "__main__":
    unittest.main()
