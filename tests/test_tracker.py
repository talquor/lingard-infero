import unittest
from algorithms.utils.tracker import BoxTracker


class TestBoxTracker(unittest.TestCase):
    def test_stable_ids_basic(self):
        tr = BoxTracker(iou_th=0.2, center_th=0.1, max_miss=3, match_by_center=True)
        ts = 0.0
        # Frame 1: two faces
        ids = tr.assign([[0.1,0.1,0.2,0.2],[0.6,0.1,0.2,0.2]], ts)
        self.assertEqual(len(ids), 2)
        id_a, id_b = ids
        self.assertNotEqual(id_a, id_b)
        ts += 0.05
        # Frame 2: slight movement
        ids2 = tr.assign([[0.11,0.1,0.2,0.2],[0.59,0.1,0.2,0.2]], ts)
        self.assertEqual(ids2[0], id_a)
        self.assertEqual(ids2[1], id_b)
        ts += 0.05
        # Frame 3: only second remains
        ids3 = tr.assign([[0.59,0.1,0.2,0.2]], ts)
        self.assertEqual(len(ids3), 1)
        self.assertEqual(ids3[0], id_b)
        ts += 0.05
        # Frame 4: first reappears near original spot
        ids4 = tr.assign([[0.11,0.11,0.2,0.2],[0.59,0.1,0.2,0.2]], ts)
        # Order may differ; check sets
        self.assertSetEqual(set(ids4), set([id_a, id_b]))


if __name__ == '__main__':
    unittest.main()

