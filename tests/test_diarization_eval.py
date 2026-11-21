import unittest

from pyannote.core import Segment

from tools.evaluate_diarization import annotation, compare


def turns(*rows):
    return annotation({"segments": [
        {"start": start, "end": end, "speaker": speaker} for start, end, speaker in rows
    ]}, "segments")


class EvaluationTests(unittest.TestCase):
    def test_split_voice_is_counted_as_an_extra_speaker_and_confusion(self):
        reference = turns((0, 2, "A"), (2, 4, "B"))
        split = turns((0, 1, "X"), (1, 2, "Y"), (2, 4, "Z"), (6, 8, "outside"))
        result = compare(reference, split, Segment(0, 4))
        self.assertEqual((result["detected"], result["expected"]), (3, 2))
        self.assertAlmostEqual(result["confusion"], 1)
        self.assertAlmostEqual(result["der"], 25)
        matched = turns((0, 2, "X"), (2, 4, "Z"))
        self.assertEqual(compare(reference, matched, Segment(0, 4))["der"], 0)

    def test_silent_hypothesis_is_missed_speech(self):
        result = compare(turns((0, 3, "A")), turns(), Segment(0, 3))
        self.assertEqual(result["detected"], 0)
        self.assertEqual(result["missed"], 3)
        self.assertEqual(result["der"], 100)
