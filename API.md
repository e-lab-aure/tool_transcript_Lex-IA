# LEX-IA - API Reference

LEX-IA is a local speech-to-text server running on a GPU machine. It exposes a
REST API over HTTP. No authentication is required (local/LAN use only).

---

## Base URL

```
http://<HOST>:5000
```

When running locally: `http://localhost:5000`
When accessed from another device on the LAN: `http://<IP_OF_GPU_MACHINE>:5000`

---

## Supported Languages

| Code   | Language              |
|--------|-----------------------|
| `auto` | Auto-detect           |
| `en`   | English               |
| `fr`   | French                |
| `de`   | German                |
| `es`   | Spanish               |
| `it`   | Italian               |
| `pt`   | Portuguese            |
| `nl`   | Dutch                 |
| `pl`   | Polish                |
| `ru`   | Russian               |
| `zh`   | Chinese               |
| `ja`   | Japanese              |
| `ko`   | Korean                |
| `ar`   | Arabic                |

---

## Transcription Models

LEX-IA uses two ASR models internally:

- **Whisper large-v3** (faster-whisper / CTranslate2) - multilingual, used for all
  live transcription and when diarization is enabled
- **Canary-Qwen-2.5B** (NVIDIA NeMo) - English only, used for longer English files
  via the async pipeline

The live endpoint always uses Whisper for lowest latency.

---

## Endpoints

### 1. `POST /api/transcribe-live` - Live transcription (synchronous)

**This is the primary endpoint for real-time/push-to-talk use cases.**

Accepts a short audio recording, transcribes it immediately, and returns the
text in the HTTP response body. The call blocks until transcription is complete.
No job ID, no polling required.

#### Request

- **Method:** `POST`
- **Content-Type:** `multipart/form-data`
- **CORS:** Enabled (`Access-Control-Allow-Origin: *`)

| Field        | Type    | Required | Default  | Description                                          |
|--------------|---------|----------|----------|------------------------------------------------------|
| `audio`      | file    | yes      | -        | Audio file binary. Any format supported by ffmpeg: WAV, MP3, WebM, OGG, M4A, FLAC, etc. |
| `language`   | string  | no       | `"auto"` | Language code from the supported list, or `"auto"` for automatic detection. |
| `preprocess` | string  | no       | `"false"` | `"true"` to apply DSP chain (high-pass filter, noise reduction, normalization). Adds ~0.5s latency. Useful for noisy environments. |

#### Response `200 OK`

```json
{
  "transcript":    "The transcribed text as a single string.",
  "language":      "fr",
  "duration_secs": 4.2
}
```

| Field           | Type   | Description                                  |
|-----------------|--------|----------------------------------------------|
| `transcript`    | string | Full transcribed text. Empty string if no speech detected. |
| `language`      | string | Detected or confirmed language code.         |
| `duration_secs` | float  | Duration of the audio that was processed (seconds). |

#### Error Responses

| HTTP Status | `error` value                              | Cause                                      |
|-------------|--------------------------------------------|--------------------------------------------|
| `400`       | `"Champ 'audio' manquant"`                 | No `audio` field in the request.           |
| `400`       | `"Langue non supportee : <value>"`         | Unknown language code.                     |
| `422`       | `"Format audio non reconnu par ffmpeg"`    | Audio file is corrupt or unsupported format. |
| `500`       | `"Erreur interne lors de la transcription"` | GPU/model error during inference.          |

#### Examples

**curl:**
```bash
# French audio, explicit language
curl -X POST http://192.168.1.10:5000/api/transcribe-live \
  -F "audio=@recording.wav" \
  -F "language=fr"

# Auto language detection
curl -X POST http://192.168.1.10:5000/api/transcribe-live \
  -F "audio=@recording.webm" \
  -F "language=auto"

# With noise reduction
curl -X POST http://192.168.1.10:5000/api/transcribe-live \
  -F "audio=@recording.mp3" \
  -F "language=fr" \
  -F "preprocess=true"
```

**Python (requests):**
```python
import requests

with open("recording.wav", "rb") as f:
    response = requests.post(
        "http://192.168.1.10:5000/api/transcribe-live",
        files={"audio": ("recording.wav", f, "audio/wav")},
        data={"language": "fr"},
    )

data = response.json()
print(data["transcript"])   # "Bonjour, voici le texte dicte."
print(data["language"])     # "fr"
print(data["duration_secs"])  # 4.2
```

**JavaScript (fetch + MediaRecorder):**
```javascript
// After recording with MediaRecorder, audioBlob is available
async function transcribe(audioBlob, language = "fr") {
  const form = new FormData();
  form.append("audio", audioBlob, "recording.webm");
  form.append("language", language);

  const response = await fetch("http://192.168.1.10:5000/api/transcribe-live", {
    method: "POST",
    body: form,
  });

  if (!response.ok) {
    const err = await response.json();
    throw new Error(err.error);
  }

  const data = await response.json();
  return data.transcript;  // string
}
```

#### Performance expectations

| Audio duration | Typical response time (RTX 3090 / 4090) |
|----------------|-----------------------------------------|
| 5 seconds      | ~1-2 seconds                            |
| 15 seconds     | ~2-4 seconds                            |
| 30 seconds     | ~4-8 seconds                            |

First request after server startup may be slower (model warm-up).
Concurrent requests are handled correctly (Flask `threaded=True`).

---

### 2. `GET /languages` - List supported languages

Returns the full map of supported language codes.

#### Request

```
GET /languages
```

No parameters.

#### Response `200 OK`

```json
{
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
  "ar":   "Arabe"
}
```

---

### 3. `POST /transcribe` - Async transcription from YouTube URL

Queues a transcription job for a YouTube video. Returns immediately with a
`job_id`. Poll `/status/<job_id>` for progress and results.

#### Request

- **Content-Type:** `application/json`

```json
{
  "url":        "https://www.youtube.com/watch?v=...",
  "language":   "auto",
  "preprocess": true,
  "diarize":    false,
  "test_mode":  0
}
```

| Field        | Type    | Required | Default | Description                                    |
|--------------|---------|----------|---------|------------------------------------------------|
| `url`        | string  | yes      | -       | YouTube video URL.                             |
| `language`   | string  | no       | `"auto"` | Language code or `"auto"`.                    |
| `preprocess` | boolean | no       | `true`  | Apply DSP audio preprocessing.                 |
| `diarize`    | boolean | no       | `false` | Enable speaker diarization (requires HF_TOKEN).|
| `test_mode`  | integer | no       | `0`     | Truncate audio to N seconds (0 = full audio).  |

#### Response `200 OK`

```json
{ "job_id": "550e8400-e29b-41d4-a716-446655440000" }
```

#### Error Response

```json
{ "error": "URL manquante" }
```

---

### 4. `POST /transcribe-file` - Async transcription from uploaded file

Queues a transcription job for a local audio/video file upload.

#### Request

- **Content-Type:** `multipart/form-data`

| Field        | Type    | Required | Default  | Description                                    |
|--------------|---------|----------|----------|------------------------------------------------|
| `file`       | file    | yes      | -        | Audio/video file (MP3, WAV, MP4, MKV, AVI, WebM, etc.). |
| `language`   | string  | no       | `"auto"` | Language code or `"auto"`.                    |
| `preprocess` | string  | no       | `"true"` | `"true"` / `"false"`.                         |
| `diarize`    | string  | no       | `"false"` | `"true"` to enable speaker diarization.      |
| `test_mode`  | string  | no       | `"0"`    | Truncate to N seconds for testing.             |

#### Response `200 OK`

```json
{ "job_id": "550e8400-e29b-41d4-a716-446655440000" }
```

---

### 5. `GET /status/<job_id>` - Poll async job status

Returns the full state of a queued or completed job.

#### Response `200 OK`

```json
{
  "status":          "done",
  "title":           "My recording.mp3",
  "transcript":      "Full transcribed text...",
  "language":        "fr",
  "model_used":      "Whisper large-v3",
  "diarized":        false,
  "audio_available": true,
  "test_mode":       0,
  "duration_secs":   42.3,
  "error":           null,
  "speakers":        [],
  "segments": [
    { "start": 0.0, "end": 3.5, "text": "Bonjour.", "speaker": null }
  ],
  "logs": [
    { "ts": "14:32:01", "msg": "Conversion en WAV...", "level": "step" }
  ]
}
```

**`status` values:**

| Value          | Meaning                                          |
|----------------|--------------------------------------------------|
| `"queued"`     | Job is waiting to start.                         |
| `"downloading"` | Downloading audio from YouTube.                 |
| `"detecting"`  | Running language detection pass.                 |
| `"transcribing"` | Model is running inference.                    |
| `"done"`       | Transcription complete. `transcript` is populated. |
| `"error"`      | Failed. See `error` field for reason.            |

**`segments`** (when Whisper was used, not Canary):
```json
[
  { "start": 0.0, "end": 3.5, "text": "First sentence.", "speaker": null },
  { "start": 3.5, "end": 7.2, "text": "Second sentence.", "speaker": "Locuteur 1" }
]
```

**`speakers`** (when `diarize=true`):
```json
[
  {
    "id": "SPEAKER_00",
    "label": "Locuteur 1",
    "clip_start": 1.2,
    "clip_end": 9.2,
    "excerpt": "First words of this speaker..."
  }
]
```

#### Error Response `404`

```json
{ "error": "Job introuvable" }
```

---

### 6. `GET /logs/<job_id>` - Get job logs

Returns the real-time activity log for an async job. Useful for showing
progress details to the user.

#### Response `200 OK`

```json
{
  "logs": [
    { "ts": "14:32:01", "msg": "Conversion en WAV 16kHz mono...", "level": "step" },
    { "ts": "14:32:02", "msg": "WAV généré : 12.4 Mo @ 16000Hz mono", "level": "success" },
    { "ts": "14:32:05", "msg": "[0.0s -> 3.5s] Bonjour.", "level": "segment" }
  ]
}
```

**`level` values:** `"step"`, `"model"`, `"segment"`, `"success"`, `"error"`, `"info"`

---

### 7. `GET /download-audio/<job_id>` - Download processed audio

Returns the processed audio as an MP3 file (only available when
`audio_available: true` in the job status).

#### Response `200 OK`
- Content-Type: `audio/mpeg`
- Content-Disposition: `attachment; filename="<title>.mp3"`

---

### 8. `GET /speaker-audio/<job_id>/<speaker_id>` - Download speaker clip

Returns a short MP3 clip (~8s) for a specific speaker, identified by their
`id` field from the `speakers` array. Only available on diarized jobs.

#### Response `200 OK`
- Content-Type: `audio/mpeg`

---

### 9. `GET /test-diarize` - Test diarization setup

Verifies that the HuggingFace token and pyannote pipeline are correctly
configured. Does not process any audio.

#### Response `200 OK`

```json
{ "ok": true, "message": "Pipeline pyannote chargé — connexion et token valides." }
```

or on failure:

```json
{ "ok": false, "error": "401 Client Error: Unauthorized..." }
```

---

## Error Format

All error responses follow the same structure:

```json
{ "error": "<human-readable message>" }
```

---

## Typical Client Workflow - Push-to-Talk

```
1. User presses record button
      -> start MediaRecorder (browser) or PyAudio stream

2. User releases record button
      -> stop recording, collect audio blob/bytes

3. POST /api/transcribe-live
      body: { audio: <blob>, language: "fr" }

4. Await response (1-8 seconds depending on duration)
      -> { transcript: "...", language: "fr", duration_secs: 5.1 }

5. Display transcript to user
```

## Typical Client Workflow - Async Long File

```
1. POST /transcribe-file   ->  { job_id: "abc-123" }

2. Poll GET /status/abc-123  every 1500ms
      while status not in ["done", "error"]

3. When status == "done"
      -> read job.transcript

4. Optionally GET /download-audio/abc-123  for the audio file
```

---

## Notes for Client Implementation

- **Audio format for browser recording:** Chrome and Firefox both produce
  `audio/webm` via `MediaRecorder`. Pass this directly - ffmpeg handles it.
- **No auth:** No API key or session token needed. The server is intended for
  LAN use only.
- **Concurrent requests:** The `/api/transcribe-live` endpoint is thread-safe.
  Multiple simultaneous requests will queue on the GPU naturally.
- **Max file size:** Flask default is unlimited; practical limit is GPU memory
  (very long audio may cause OOM on `/api/transcribe-live`). For files longer
  than ~5 minutes, prefer the async `/transcribe-file` endpoint.
- **Cold start:** The first transcription request after server startup will be
  ~2-5s slower as the GPU warms up.
- **Server address:** LEX-IA listens on `0.0.0.0:5000`. Find the host machine's
  LAN IP with `ipconfig` (Windows) or `ip addr` (Linux).
