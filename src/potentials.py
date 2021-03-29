from copy import Error
from typing import Dict, List, Tuple, Union
import numpy as np
import json


class Potentials:
    def __init__(self, adjacent_weight=1/6, diagonal_weight=1/12) -> None:
        self._adjacent_weight = adjacent_weight
        self._diagonal_weight = diagonal_weight
        self._model: Dict[str, List[List[int]]] = {}

    def fit(self, data: Dict[str, List[np.ndarray]]):
        for label, images in data.items():
            self._model[label] = []

            for image in images:
                potential = self._potential(image)
                self._model[label].append(potential)

    def predict(self, data: List[np.ndarray]) -> List[str]:
        if len(self._model) == 0:
            raise Error('Model is empty; train or load it from file')

        labels = []
        for item in data:
            label = self._predict_item(item)
            labels.append(label)

        return labels

    def save_model(self, path: str) -> None:
        s = json.dumps(self._model, indent=2)
        with open(path, "w") as f:
            f.write(s)

    def load_model(self, path: str) -> None:
        with open(path) as f:
            s = f.read()
            self._model = json.loads(s)

    def _predict_item(self, item: np.ndarray) -> str:
        potential = self._potential(item)

        min_distance = -1
        matched_label = ''

        for label, images in self._model.items():
            for image in images:
                distance = self._distance(image, potential)

                if min_distance < 0 or distance < min_distance:
                    min_distance = distance
                    matched_label = label

        return matched_label

    def _distance(self, x: List[int], y: List[int]) -> float:
        return abs(sum([x - y for x, y in zip(x, y)]))

    def _potential(self, img: np.ndarray) -> List[int]:
        img = img.astype('int32')
        results = []

        for row in range(img.shape[0]):
            for col in range(img.shape[1]):
                value = self._cell_value(img, row, col)
                results.append(value)

        return results

    def _cell_value(self, img: np.ndarray, row: int, col: int) -> int:
        adjacent = self._safe(img, row, col-1) + self._safe(img, row, col+1) + \
            self._safe(img, row-1, col) + self._safe(img, row+1, col)
        diagonal = self._safe(img, row-1, col-1) + self._safe(img, row-1, col+1) + \
            self._safe(img, row+1, col-1) + self._safe(img, row+1, col+1)
        return self._safe(img, row, col) + self._adjacent_weight * adjacent + self._diagonal_weight * diagonal

    def _safe(self, img: np.ndarray, row: int, col: int) -> int:
        if row < 0 or col < 0 or row >= img.shape[0] or col >= img.shape[1]:
            return 0
        else:
            return int(img[row, col] > 0)
