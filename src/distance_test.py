import random
import unittest

from scipy.spatial import distance as scipy

import distance


class DistanceTest(unittest.TestCase):
    @staticmethod
    def random_lists():
        def random_list():
            return [random.randint(-100, 100) for _ in range(100)]

        return random_list(), random_list()

    def test_euclidean(self):
        for _ in range(100):
            x, y = self.random_lists()

            expected = scipy.euclidean(x, y)
            actual = distance.euclidean(x, y)
            self.assertAlmostEqual(expected, actual)

    def test_hamming(self):
        for _ in range(100):
            x, y = self.random_lists()

            expected = scipy.hamming(x, y)
            actual = distance.hamming(x, y)
            self.assertAlmostEqual(expected, actual)

    def test_manhattan(self):
        for _ in range(100):
            x, y = self.random_lists()

            expected = scipy.cityblock(x, y)
            actual = distance.manhattan(x, y)
            self.assertAlmostEqual(expected, actual)

    def test_jaccard(self):
        for _ in range(100):
            x, y = self.random_lists()

            # noinspection PyTypeChecker
            expected = scipy.jaccard(x, y)
            actual = distance.jaccard(x, y)
            self.assertAlmostEqual(expected, actual)

    def test_cosine(self):
        for _ in range(100):
            x, y = self.random_lists()

            expected = scipy.cosine(x, y)
            actual = distance.cosine(x, y)
            self.assertAlmostEqual(expected, actual)


if __name__ == '__main__':
    unittest.main()
