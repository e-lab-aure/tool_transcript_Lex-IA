"""Compare des exports LEX-IA à une annotation de référence, hors ligne."""

import argparse
import json
import math
from pathlib import Path

from pyannote.core import Annotation, Segment, Timeline
from pyannote.metrics.diarization import DiarizationErrorRate


def seconds(value, field):
    if type(value) not in (int, float) or not 0 <= value < math.inf:
        raise ValueError(f"{field} doit être un nombre de secondes positif ou nul.")
    return value


def annotation(data, field):
    rows = data.get(field)
    if not isinstance(rows, list):
        raise ValueError(f"La liste {field} est manquante.")
    result = Annotation()
    for track, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError(f"Passage {track} invalide.")
        start = seconds(row.get("start"), f"Début du passage {track}")
        end = seconds(row.get("end"), f"Fin du passage {track}")
        if end == start:
            continue
        if end < start:
            raise ValueError(f"Fin avant le début au passage {track}.")
        speaker = row.get("speaker_id", row.get("speaker"))
        if speaker is None or speaker == "":
            continue
        if not isinstance(speaker, str):
            raise ValueError(f"Locuteur invalide au passage {track}.")
        result[Segment(start, end), track] = speaker
    return result


def compare(reference, prediction, window):
    support = Timeline([window])
    reference = reference.crop(support, mode="intersection")
    prediction = prediction.crop(support, mode="intersection")
    if not reference:
        raise ValueError("La fenêtre ne contient aucun passage annoté dans la référence.")
    detail = DiarizationErrorRate(collar=0, skip_overlap=False)(
        reference, prediction, uem=support, detailed=True,
    )
    return {
        "expected": len(reference.labels()),
        "detected": len(prediction.labels()),
        "der": 100 * detail["diarization error rate"],
        "confusion": detail["confusion"],
        "missed": detail["missed detection"],
        "false_alarm": detail["false alarm"],
    }


def read_json(path):
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Lecture impossible de {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Un objet JSON est attendu dans {path}.")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path, help="Annotation vérifiée à l'écoute")
    parser.add_argument("exports", type=Path, nargs="+", help="Exports JSON à comparer")
    parser.add_argument("--start", type=float, default=0, help="Début de la fenêtre en secondes")
    parser.add_argument("--end", type=float, help="Fin de la fenêtre en secondes")
    parser.add_argument("--edited", action="store_true", help="Évalue les passages du lecteur après correction")
    parser.add_argument("--correction-seconds", type=float, action="append", metavar="S",
                        help="Temps manuel mesuré. Répéter une fois par export, dans le même ordre")
    args = parser.parse_args()
    try:
        correction_times = args.correction_seconds
        if correction_times is not None and len(correction_times) != len(args.exports):
            raise ValueError("Indiquez un temps de correction pour chaque export.")
        if correction_times:
            correction_times = [seconds(value, "Temps de correction") for value in correction_times]

        data = read_json(args.reference)
        reference = annotation(data, "segments")
        start = seconds(args.start, "Début de la fenêtre")
        end = args.end if args.end is not None else data.get("duration_secs", reference.get_timeline().extent().end)
        end = seconds(end, "Fin de la fenêtre")
        if end <= start:
            raise ValueError("La fenêtre d’évaluation est vide.")

        field = "segments" if args.edited else "diarization_turns"
        for index, path in enumerate(args.exports):
            data = read_json(path)
            if not args.edited and not data.get("diarized", bool(data.get("diarization_turns"))):
                raise ValueError(f"{path} ne contient aucun tour de diarisation. Utilisez un export diarizé ou --edited.")
            result = compare(reference, annotation(data, field), Segment(start, end))
            row = (
                f"{path.name} | voix {result['detected']}/{result['expected']} "
                f"| DER {result['der']:.1f} % | confusion {result['confusion']:.2f} s "
                f"| parole manquée {result['missed']:.2f} s | fausse alerte {result['false_alarm']:.2f} s"
            )
            if correction_times:
                row += f" | correction {correction_times[index]:.1f} s"
            print(row)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
