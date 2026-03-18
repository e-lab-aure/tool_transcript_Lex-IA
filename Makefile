.PHONY: install run clean update

VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

install:
	@echo "→ Création du virtualenv..."
	python3 -m venv $(VENV)
	@echo "→ Installation des dépendances..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "✅ Installation terminée. Lance 'make run' pour démarrer."

run:
	@echo "→ Démarrage du serveur sur http://localhost:5000"
	$(PYTHON) app.py

update:
	@echo "→ Mise à jour de yt-dlp..."
	$(PIP) install --upgrade yt-dlp
	@echo "✅ yt-dlp mis à jour."

clean:
	@echo "→ Suppression du virtualenv..."
	rm -rf $(VENV)
	@echo "✅ Nettoyage terminé."
