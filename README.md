# LEX-IA

Transcripteur audio/vidéo local propulsé par GPU NVIDIA.

## Stack
- **Canary-Qwen-2.5B** (NeMo) — anglais, chunking 40s
- **Whisper large-v3** (faster-whisper) — toutes les autres langues
- **yt-dlp** + **deno** — extraction audio/vidéo (deno résout les challenges JS de YouTube, installé dans le venv)
- **ffmpeg** — conversion audio
- **Flask** — serveur web local

## Prérequis
- Python 3.11 ou 3.12
- GPU NVIDIA avec CUDA 12.8
- ffmpeg dans le PATH
- Git (pour l'installation de NeMo)

## Installation
```bat
install.bat
```

## Lancement

### Icône de zone de notification (recommandé)
```bat
tray.bat
```
Une icône apparaît près de l'horloge et pilote le serveur sans console.

Menu (clic droit) :
- **Ouvrir l'interface** — http://localhost:5000 (double-clic sur l'icône)
- **Démarrer** / **Suspendre** / **Reprendre** / **Redémarrer** / **Arrêter**
- **Ouvrir le journal** — sortie du serveur dans `logs/server.log`
- **Démarrer avec Windows** — crée le raccourci de démarrage de session
- **Démarrer le serveur automatiquement** — lance le serveur avec l'icône

Couleur de l'icône : gris = arrêté, bleu = démarrage, vert = prêt, orange = suspendu, rouge = erreur.

Le serveur ne répond qu'après le chargement des deux modèles, ce qui prend une à
deux minutes. **Suspendre** gèle le processus mais garde la VRAM allouée ; seul
**Arrêter** décharge les modèles du GPU.

### Console
```bat
run.bat
```

## Démarrage automatique de Windows

Coche **Démarrer avec Windows** dans le menu de l'icône. Un raccourci est créé
dans le dossier de démarrage de la session, sans droits administrateur. Décoche
pour le retirer.

## Mise à jour yt-dlp
yt-dlp (avec deno et yt-dlp-ejs) est mis à jour automatiquement à chaque démarrage du serveur (avant le
chargement des modèles). En cas d'échec (hors ligne), la version installée est
conservée. Pour désactiver : `LEXIA_SKIP_YTDLP_UPDATE=1` dans `.env`.

Mise à jour manuelle :
```bat
update.bat
```

## Architecture

![Architecture LEX-IA](docs/lex_ia_architecture.svg)

## Fonctionnalités
- URL YouTube ou fichier local (drag & drop)
- Onglet **Vidéo** : téléchargement de la vidéo complète (yt-dlp, MP4, qualité au choix)
- Détection automatique de la langue
- Routage intelligent vers le meilleur modèle
- Console IA en temps réel
- Export transcript en .txt
