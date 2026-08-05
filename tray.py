"""Icone de zone de notification pour piloter le serveur LEX-IA sous Windows.

Lance app.py dans un sous-processus sans console et expose un menu permettant de
demarrer, suspendre, reprendre, redemarrer et arreter le serveur, ainsi que
d'activer le lancement automatique a l'ouverture de session.

Note : suspendre gele le processus mais ne libere pas la VRAM. Seul l'arret
decharge les modeles du GPU.
"""

import ctypes
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from datetime import datetime

import psutil
import pystray
from PIL import Image, ImageDraw

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
VENV_PYTHON  = os.path.join(BASE_DIR, ".venv", "Scripts", "python.exe")
VENV_PYTHONW = os.path.join(BASE_DIR, ".venv", "Scripts", "pythonw.exe")
APP_SCRIPT   = os.path.join(BASE_DIR, "app.py")
TRAY_SCRIPT  = os.path.abspath(__file__)
LOG_DIR      = os.path.join(BASE_DIR, "logs")
LOG_FILE     = os.path.join(LOG_DIR, "server.log")
CONFIG_FILE  = os.path.join(BASE_DIR, "tray_config.json")

SERVER_URL = "http://127.0.0.1:5000"
HEALTH_URL = f"{SERVER_URL}/health"

SHORTCUT_NAME = "LEX-IA.lnk"
MUTEX_NAME    = "Local\\LEX-IA-tray"

MAX_LOG_BYTES    = 5 * 1024 * 1024
POLL_INTERVAL    = 2.0     # secondes entre deux verifications d'etat
HEALTH_TIMEOUT   = 3.0     # secondes avant abandon d'une requete /health
START_TIMEOUT    = 900.0   # au-dela, le chargement des modeles est considere echoue
STOP_TIMEOUT     = 20.0    # delai avant de forcer la fin du processus

CREATE_NO_WINDOW = 0x08000000

# Etats du serveur
STOPPED  = "stopped"
STARTING = "starting"
RUNNING  = "running"
PAUSED   = "paused"
STOPPING = "stopping"
FAILED   = "failed"

STATE_LABELS = {
    STOPPED:  "Arrete",
    STARTING: "Demarrage (chargement des modeles)...",
    RUNNING:  "En cours",
    PAUSED:   "Suspendu",
    STOPPING: "Arret en cours...",
    FAILED:   "Arrete apres une erreur",
}

STATE_COLORS = {
    STOPPED:  (97, 97, 97),
    STARTING: (21, 101, 192),
    RUNNING:  (46, 125, 50),
    PAUSED:   (239, 108, 0),
    STOPPING: (21, 101, 192),
    FAILED:   (198, 40, 40),
}

# Constantes MessageBox de l'API Windows
MB_YESNO       = 0x00000004
MB_ICONWARNING = 0x00000030
MB_ICONERROR   = 0x00000010
MB_TOPMOST     = 0x00040000
IDYES          = 6


# ---------------------------------------------------------------------------
# Helpers Windows
# ---------------------------------------------------------------------------

def _message_box(title: str, text: str, flags: int) -> int:
    return ctypes.windll.user32.MessageBoxW(None, text, title, flags)


def _confirm(title: str, text: str) -> bool:
    return _message_box(title, text, MB_YESNO | MB_ICONWARNING | MB_TOPMOST) == IDYES


def _error_box(title: str, text: str) -> None:
    _message_box(title, text, MB_ICONERROR | MB_TOPMOST)


def _ps_quote(value: str) -> str:
    """Echappe une valeur pour une chaine PowerShell entre quotes simples."""
    return "'" + value.replace("'", "''") + "'"


def _acquire_single_instance() -> bool:
    """Empeche deux icones de piloter deux serveurs sur le meme port.

    Le mutex reste detenu tant que le processus vit ; le handle n'est donc
    jamais libere explicitement.
    """
    ERROR_ALREADY_EXISTS = 183
    handle = ctypes.windll.kernel32.CreateMutexW(None, False, MUTEX_NAME)
    if not handle:
        return True  # en cas d'echec de l'API, ne pas bloquer le lancement
    return ctypes.windll.kernel32.GetLastError() != ERROR_ALREADY_EXISTS


def _startup_shortcut_path() -> str:
    appdata = os.environ.get("APPDATA", "")
    return os.path.join(
        appdata, "Microsoft", "Windows", "Start Menu", "Programs", "Startup", SHORTCUT_NAME
    )


# ---------------------------------------------------------------------------
# Configuration persistante
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {"autostart_server": True}


def load_config() -> dict:
    config = dict(DEFAULT_CONFIG)
    try:
        with open(CONFIG_FILE, encoding="utf-8") as fh:
            stored = json.load(fh)
        if isinstance(stored, dict):
            for key in DEFAULT_CONFIG:
                if key in stored:
                    config[key] = bool(stored[key])
    except FileNotFoundError:
        pass
    except (OSError, ValueError) as exc:
        print(f"[tray] Configuration illisible, valeurs par defaut utilisees : {exc}")
    return config


def save_config(config: dict) -> None:
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as fh:
            json.dump(config, fh, indent=2)
    except OSError as exc:
        print(f"[tray] Impossible d'enregistrer la configuration : {exc}")


# ---------------------------------------------------------------------------
# Icone
# ---------------------------------------------------------------------------

def build_icon(color: tuple) -> Image.Image:
    """Dessine un micro blanc sur une pastille coloree indiquant l'etat."""
    size  = 64
    white = (255, 255, 255, 255)
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw  = ImageDraw.Draw(image)

    draw.rounded_rectangle([0, 0, size - 1, size - 1], radius=14, fill=color + (255,))
    draw.rounded_rectangle([25, 12, 39, 38], radius=7, fill=white)
    draw.arc([18, 20, 46, 46], start=0, end=180, fill=white, width=5)
    draw.line([32, 46, 32, 53], fill=white, width=5)
    draw.line([23, 53, 41, 53], fill=white, width=5)
    return image


ICONS = {state: build_icon(color) for state, color in STATE_COLORS.items()}


# ---------------------------------------------------------------------------
# Controleur du serveur
# ---------------------------------------------------------------------------

class ServerController:
    """Pilote le sous-processus app.py et suit son etat."""

    def __init__(self, on_change) -> None:
        self._lock       = threading.RLock()
        self._proc       = None
        self._log_handle = None
        self._state      = STOPPED
        self._started_at = 0.0
        self._on_change  = on_change

    # -- Etat ---------------------------------------------------------------

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    def _set_state(self, state: str) -> None:
        with self._lock:
            if self._state == state:
                return
            self._state = state
        self._on_change()

    def is_active(self) -> bool:
        return self.state in (STARTING, RUNNING, PAUSED)

    # -- Journal ------------------------------------------------------------

    def _open_log(self):
        os.makedirs(LOG_DIR, exist_ok=True)
        # Repart d'un fichier vide si le journal est devenu trop volumineux.
        mode = "a"
        if os.path.exists(LOG_FILE) and os.path.getsize(LOG_FILE) > MAX_LOG_BYTES:
            mode = "w"
        handle = open(LOG_FILE, mode, encoding="utf-8")
        handle.write(f"\n===== Demarrage {datetime.now():%Y-%m-%d %H:%M:%S} =====\n")
        handle.flush()
        return handle

    def _close_log(self) -> None:
        if self._log_handle is not None:
            try:
                self._log_handle.close()
            except OSError:
                pass
            self._log_handle = None

    # -- Cycle de vie -------------------------------------------------------

    def start(self) -> None:
        with self._lock:
            if self.is_active():
                return
            if not os.path.isfile(VENV_PYTHON):
                _error_box(
                    "LEX-IA",
                    "Le virtualenv est introuvable.\n\n"
                    f"Attendu : {VENV_PYTHON}\n\n"
                    "Lance install.bat avant d'utiliser l'icone.",
                )
                self._set_state(FAILED)
                return

            try:
                self._log_handle = self._open_log()
            except OSError as exc:
                _error_box("LEX-IA", f"Impossible d'ecrire le journal :\n{exc}")
                self._set_state(FAILED)
                return

            env = os.environ.copy()
            # Le serveur affiche des emojis : sans UTF-8 force, l'ecriture dans
            # le fichier journal leverait UnicodeEncodeError et tuerait app.py.
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUNBUFFERED"] = "1"

            try:
                self._proc = subprocess.Popen(
                    [VENV_PYTHON, APP_SCRIPT],
                    cwd=BASE_DIR,
                    stdout=self._log_handle,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    env=env,
                    creationflags=CREATE_NO_WINDOW,
                )
            except OSError as exc:
                self._close_log()
                _error_box("LEX-IA", f"Demarrage impossible :\n{exc}")
                self._set_state(FAILED)
                return

            self._started_at = time.monotonic()

        self._set_state(STARTING)

    def stop(self) -> None:
        with self._lock:
            proc = self._proc
            if proc is None:
                self._set_state(STOPPED)
                return

        self._set_state(STOPPING)
        self._terminate_tree(proc)

        with self._lock:
            self._proc = None
            self._close_log()
        self._set_state(STOPPED)

    def _terminate_tree(self, proc: subprocess.Popen) -> None:
        """Termine app.py et ses enfants (ffmpeg, yt-dlp) proprement."""
        try:
            parent = psutil.Process(proc.pid)
        except psutil.NoSuchProcess:
            proc.poll()
            return

        try:
            targets = parent.children(recursive=True) + [parent]
        except psutil.Error:
            targets = [parent]

        for target in targets:
            try:
                # Un processus gele ignore la demande d'arret : le reveiller d'abord.
                if target.status() == psutil.STATUS_STOPPED:
                    target.resume()
            except psutil.Error:
                pass
            try:
                target.terminate()
            except psutil.Error:
                pass

        _, alive = psutil.wait_procs(targets, timeout=STOP_TIMEOUT)
        for target in alive:
            try:
                target.kill()
            except psutil.Error:
                pass
        psutil.wait_procs(alive, timeout=5)
        proc.poll()

    def restart(self) -> None:
        self.stop()
        self.start()

    # -- Suspension ---------------------------------------------------------

    def pause(self) -> None:
        with self._lock:
            if self._state != RUNNING or self._proc is None:
                return
            if not self._apply_to_tree("suspend"):
                return
        self._set_state(PAUSED)

    def resume(self) -> None:
        with self._lock:
            if self._state != PAUSED or self._proc is None:
                return
            if not self._apply_to_tree("resume"):
                return
        self._set_state(RUNNING)

    def _apply_to_tree(self, action: str) -> bool:
        """Applique suspend/resume au processus serveur et a ses enfants."""
        try:
            parent  = psutil.Process(self._proc.pid)
            targets = parent.children(recursive=True) + [parent]
        except psutil.Error as exc:
            print(f"[tray] {action} impossible : {exc}")
            return False

        applied = False
        for target in targets:
            try:
                getattr(target, action)()
                applied = True
            except psutil.Error as exc:
                print(f"[tray] {action} ignore pour le PID {target.pid} : {exc}")
        return applied

    # -- Surveillance -------------------------------------------------------

    def health(self) -> dict | None:
        """Interroge /health ; renvoie None si le serveur ne repond pas."""
        try:
            with urllib.request.urlopen(HEALTH_URL, timeout=HEALTH_TIMEOUT) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, ValueError, TimeoutError):
            return None

    def active_jobs(self) -> int:
        data = self.health()
        if not data:
            return 0
        try:
            return int(data.get("active_jobs", 0))
        except (TypeError, ValueError):
            return 0

    def poll(self) -> str | None:
        """Rafraichit l'etat. Renvoie un message a notifier, le cas echeant."""
        with self._lock:
            state = self._state
            proc  = self._proc
            if state in (STOPPED, STOPPING, FAILED) or proc is None:
                return None

            exit_code = proc.poll()
            if exit_code is not None:
                self._proc = None
                self._close_log()
                self._state = STOPPED if exit_code == 0 else FAILED
                self._on_change()
                if exit_code == 0:
                    return "Le serveur s'est arrete."
                return f"Le serveur s'est arrete (code {exit_code}). Consulte le journal."

            if state != STARTING:
                return None
            elapsed = time.monotonic() - self._started_at

        # Etat STARTING : le serveur ne repond qu'une fois les modeles charges.
        # La requete reseau est faite hors verrou pour ne pas bloquer le menu.
        if self.health() is not None:
            self._set_state(RUNNING)
            return "Serveur pret sur http://localhost:5000"
        if elapsed > START_TIMEOUT:
            self.stop()
            self._set_state(FAILED)
            return "Demarrage trop long, serveur arrete. Consulte le journal."
        return None


# ---------------------------------------------------------------------------
# Application de la zone de notification
# ---------------------------------------------------------------------------

class TrayApp:
    def __init__(self) -> None:
        self.config     = load_config()
        self.server     = ServerController(self._refresh_icon)
        self._stopping  = threading.Event()
        self.icon       = pystray.Icon(
            "lex-ia",
            ICONS[STOPPED],
            "LEX-IA - arrete",
            menu=self._build_menu(),
        )

    # -- Menu ---------------------------------------------------------------

    def _build_menu(self) -> pystray.Menu:
        return pystray.Menu(
            pystray.MenuItem(
                lambda _: f"LEX-IA : {STATE_LABELS[self.server.state]}",
                lambda *_: None,
                enabled=False,
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(
                "Ouvrir l'interface",
                self._on_open,
                default=True,
                enabled=lambda _: self.server.state == RUNNING,
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(
                "Demarrer",
                self._on_start,
                enabled=lambda _: self.server.state in (STOPPED, FAILED),
            ),
            pystray.MenuItem(
                lambda _: "Reprendre" if self.server.state == PAUSED else "Suspendre",
                self._on_toggle_pause,
                enabled=lambda _: self.server.state in (RUNNING, PAUSED),
            ),
            pystray.MenuItem(
                "Redemarrer",
                self._on_restart,
                enabled=lambda _: self.server.is_active(),
            ),
            pystray.MenuItem(
                "Arreter",
                self._on_stop,
                enabled=lambda _: self.server.is_active(),
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Ouvrir le journal", self._on_open_log),
            pystray.MenuItem(
                "Demarrer avec Windows",
                self._on_toggle_startup,
                checked=lambda _: os.path.isfile(_startup_shortcut_path()),
            ),
            pystray.MenuItem(
                "Demarrer le serveur automatiquement",
                self._on_toggle_autostart,
                checked=lambda _: self.config["autostart_server"],
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quitter", self._on_quit),
        )

    def _refresh_icon(self) -> None:
        state = self.server.state
        self.icon.icon  = ICONS[state]
        self.icon.title = f"LEX-IA - {STATE_LABELS[state]}"
        self.icon.update_menu()

    def _notify(self, message: str) -> None:
        try:
            self.icon.notify(message, "LEX-IA")
        except Exception as exc:  # notification non critique selon le backend
            print(f"[tray] Notification impossible : {exc}")

    # -- Actions du menu ----------------------------------------------------

    def _on_open(self, *_) -> None:
        webbrowser.open(SERVER_URL)

    def _on_start(self, *_) -> None:
        self.server.start()

    def _on_toggle_pause(self, *_) -> None:
        if self.server.state == PAUSED:
            self.server.resume()
        else:
            self.server.pause()

    def _on_restart(self, *_) -> None:
        if self._confirm_interrupt("redemarrer"):
            self.server.restart()

    def _on_stop(self, *_) -> None:
        if self._confirm_interrupt("arreter"):
            self.server.stop()

    def _confirm_interrupt(self, action: str) -> bool:
        """Previent si une transcription est en cours avant de couper le serveur."""
        count = self.server.active_jobs()
        if count <= 0:
            return True
        return _confirm(
            "LEX-IA",
            f"{count} transcription(s) en cours seront perdues.\n\n"
            f"Veux-tu quand meme {action} le serveur ?",
        )

    def _on_open_log(self, *_) -> None:
        if not os.path.isfile(LOG_FILE):
            _error_box("LEX-IA", "Aucun journal disponible : le serveur n'a jamais demarre.")
            return
        try:
            os.startfile(LOG_FILE)
        except OSError as exc:
            _error_box("LEX-IA", f"Ouverture du journal impossible :\n{exc}")

    def _on_toggle_startup(self, *_) -> None:
        if os.path.isfile(_startup_shortcut_path()):
            self._remove_startup_shortcut()
        else:
            self._create_startup_shortcut()
        self.icon.update_menu()

    def _on_toggle_autostart(self, *_) -> None:
        self.config["autostart_server"] = not self.config["autostart_server"]
        save_config(self.config)
        self.icon.update_menu()

    def _on_quit(self, *_) -> None:
        if self.server.is_active() and not self._confirm_interrupt("quitter et arreter"):
            return
        self._stopping.set()
        self.server.stop()
        self.icon.stop()

    # -- Raccourci de demarrage --------------------------------------------

    def _create_startup_shortcut(self) -> None:
        lnk       = _startup_shortcut_path()
        target    = VENV_PYTHONW if os.path.isfile(VENV_PYTHONW) else VENV_PYTHON
        arguments = '"' + TRAY_SCRIPT + '"'
        script = (
            f"$s = (New-Object -ComObject WScript.Shell).CreateShortcut({_ps_quote(lnk)});"
            f"$s.TargetPath = {_ps_quote(target)};"
            f"$s.Arguments = {_ps_quote(arguments)};"
            f"$s.WorkingDirectory = {_ps_quote(BASE_DIR)};"
            "$s.Description = 'Serveur de transcription LEX-IA';"
            "$s.Save()"
        )
        try:
            subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                 "-Command", script],
                check=True,
                capture_output=True,
                creationflags=CREATE_NO_WINDOW,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            _error_box("LEX-IA", f"Creation du raccourci de demarrage impossible :\n{exc}")
            return
        self._notify("LEX-IA demarrera a l'ouverture de session.")

    def _remove_startup_shortcut(self) -> None:
        try:
            os.remove(_startup_shortcut_path())
        except OSError as exc:
            _error_box("LEX-IA", f"Suppression du raccourci impossible :\n{exc}")
            return
        self._notify("Demarrage automatique desactive.")

    # -- Boucle de surveillance --------------------------------------------

    def _monitor(self) -> None:
        while not self._stopping.wait(POLL_INTERVAL):
            try:
                message = self.server.poll()
            except Exception as exc:  # la boucle ne doit jamais mourir
                print(f"[tray] Erreur de surveillance : {exc}")
                continue
            if message:
                self._notify(message)

    def _setup(self, icon) -> None:
        icon.visible = True
        self._refresh_icon()
        threading.Thread(target=self._monitor, daemon=True).start()
        if self.config["autostart_server"]:
            self.server.start()

    def run(self) -> None:
        self.icon.run(setup=self._setup)
        self._stopping.set()
        self.server.stop()


def main() -> int:
    if sys.platform != "win32":
        print("Ce lanceur cible Windows.")
        return 1
    if not _acquire_single_instance():
        _error_box("LEX-IA", "LEX-IA est deja lance : son icone se trouve dans la zone de notification.")
        return 1
    TrayApp().run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
