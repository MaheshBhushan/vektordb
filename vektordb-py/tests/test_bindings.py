import tempfile
import unittest

import numpy as np

import vektordb


class BindingBoundaryTests(unittest.TestCase):
    def test_zero_k_returns_empty_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            db = vektordb.VektorDb(directory, 4)
            queries = np.zeros((3, 4), dtype=np.float32)
            ids, distances = db.search(queries, 0)
            self.assertEqual(ids.shape, (3, 0))
            self.assertEqual(distances.shape, (3, 0))
            pq_ids, pq_distances = db.search_pq(queries, 0)
            self.assertEqual(pq_ids.shape, (3, 0))
            self.assertEqual(pq_distances.shape, (3, 0))

    def test_invalid_hnsw_m_is_a_value_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                vektordb.VektorDb(directory, 4, m=1)


if __name__ == "__main__":
    unittest.main()
