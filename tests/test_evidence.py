import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from audit import audit


class EvidenceTests(unittest.TestCase):
    def test_original_measurements_replay(self):
        report=audit()
        self.assertEqual(report['decay']['approximately_inverse_distance'],101)
        self.assertGreaterEqual(len(report['replayed_conditions']),25)


if __name__=='__main__':
    unittest.main()
