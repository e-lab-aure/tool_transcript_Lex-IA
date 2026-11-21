# Évaluer la diarisation sur ses enregistrements

Choisis quelques extraits où tu connais les personnes présentes. Prévois un
entretien calme à deux, une conversation avec des interruptions et une vidéo
YouTube où les micros ou les intervenants changent. Une minute annotée avec
soin vaut mieux qu'une longue transcription simplement relue.

1. Transcris avec **Distinguer les personnes qui parlent** en mode **Autodétection**, puis exporte le
   résultat JSON depuis le lecteur. Nomme le fichier `automatique.json`.
2. Sur le même travail, relance la diarisation avec le **Nombre exact** si tu
   le connais, ou le **Nombre maximal** si tu ne connais qu'une borne.
   Exporte le nouveau JSON, par exemple `exact.json`.
3. À l'écoute, note le début et la fin de chaque prise de parole dans la
   fenêtre choisie. Utilise un identifiant stable par personne. Laisse les
   silences sans annotation. Si deux personnes parlent ensemble, note les
   deux intervalles. Pour une mesure indépendante, vérifie les débuts et fins
   à l'écoute, même si tu pars des timestamps de l'interface.
4. Compare les variantes sur la même fenêtre et sur la même piste audio.
   Garde les timestamps de la vidéo d'origine si la fenêtre commence en
   cours de vidéo. Chronomètre séparément la correction manuelle si tu veux
   mesurer le gain pratique dans le lecteur.

Exemple de référence fictive pour les huit premières secondes. L'intervalle de
4,1 à 4,5 secondes contient deux voix, et le reste de la fenêtre est silencieux.

```json
{
  "duration_secs": 8,
  "segments": [
    {"start": 0.4, "end": 2.9, "speaker": "A"},
    {"start": 3.1, "end": 4.5, "speaker": "B"},
    {"start": 4.1, "end": 6.2, "speaker": "A"}
  ]
}
```

```bat
.venv\Scripts\python.exe tools\evaluate_diarization.py reference.json automatique.json exact.json
```

Pour n'évaluer que les secondes 120 à 180, ajoute `--start 120 --end 180`.
Pour comparer le temps de correction mesuré pour deux variantes, ajoute
`--correction-seconds 50 --correction-seconds 31`, dans l'ordre des exports.
Le script lit les fichiers localement et n'utilise pas le GPU.

Le rapport affiche les voix détectées et attendues, puis le taux d'erreur de
diarisation (DER). Il compte les secondes de confusion, de parole manquée et
de fausse alerte. Les identifiants sont appariés avant le calcul, donc changer
« Locuteur 1 » en « Alice » ne réduit pas l'erreur à lui seul. Les chevauchements
de la référence sont comptés. Compare surtout les variantes sur la même fenêtre.

Par défaut le rapport évalue les tours exclusifs renvoyés par Community-1.
Ajoute `--edited` pour évaluer les passages du lecteur, avec les réattributions
et fusions exportées. Cette mesure dépend aussi de la segmentation de Whisper.
Garde le fichier de référence et les exports de chaque variante si tu veux
répéter la comparaison après une mise à jour.
