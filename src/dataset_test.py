import random
import unittest

import math

import dataset


class LoadTest(unittest.TestCase):
    def test_load_iris(self):
        head, x, labels = dataset.from_file('../data/iris.csv')
        self.assertEqual(5, head.size)
        self.assertEqual(150, len(x))
        self.assertEqual(150, len(labels))

    def test_load_fruit_and_vegetables_train(self):
        head, x, labels = dataset.from_file('../data/fruits-and-vegetables-train.csv')
        self.assertEqual(11, head.size)
        self.assertEqual(35, len(x))
        self.assertEqual(35, len(labels))

    def test_load_fruit_and_vegetables_test(self):
        head, x, labels = dataset.from_file('../data/fruits-and-vegetables-test.csv')
        self.assertEqual(10, head.size)
        self.assertEqual(4, len(x))
        self.assertEqual(4, len(labels))


class SplitTest(unittest.TestCase):
    def test_split(self):
        for i in range(100):
            ratio = random.randint(5, 95) / 100
            x = [random.randint(-100, 100) for _ in range(random.randint(2, 20))]
            labels = [random.randint(-100, 100) for _ in range(len(x))]

            (x1, labels1), (x2, labels2) = dataset.split(x, labels, ratio)

            self.assertTrue(math.floor(len(x) * ratio) <= len(x1) <= math.ceil(len(x) * ratio))
            self.assertTrue(math.floor(len(labels) * ratio) <= len(labels1) <= math.ceil(len(labels) * ratio))

            x1.extend(x2)
            labels1.extend(labels2)

            self.assertListEqual(x, x1)
            self.assertListEqual(labels, labels1)


class ShuffleTest(unittest.TestCase):
    def test_shuffle(self):
        for _ in range(100):
            ratio = random.randint(5, 95) / 100
            x = [random.randint(-100, 100) for _ in range(random.randint(2, 20))]
            labels = [random.randint(-100, 100) for _ in range(len(x))]

            shuffled_x, shuffled_labels = dataset.shuffle(x, labels)

            self.assertEqual(len(x), len(shuffled_x))
            self.assertEqual(len(labels), len(shuffled_labels))


if __name__ == '__main__':
    unittest.main()
