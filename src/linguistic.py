from copy import Error
import re
from typing import Dict, List, Tuple, Union
import numpy as np
import json
import jellyfish


class Signature:
    pass


class Signature:
    def __init__(self, code: int = None) -> None:
        self._code: int = code
        self._children: List[Signature] = []

    def add(self, s: Union[Signature, int]):
        self._children.append(s)

    def __str__(self) -> str:
        if len(self._children) > 1:
            s = map(lambda x: f'({x})', self._children)
            s = ''.join(map(str, s))
        elif len(self._children) == 1:
            child = self._children[0]
            s = f'({child})' if not self._code else str(
                child) if child._code else ''
        else:
            s = str(self._code)

        return f'{self._code}{s}' if self._code else f'{s}'


class Linguistic:
    def __init__(self) -> None:
        self._model: Dict[str, List[str]] = {}

    def fit(self, data: Dict[str, List[np.ndarray]]):
        for label, images in data.items():
            self._model[label] = []

            for image in images:
                signature = self._signature(image)
                normalized = self._normalize(signature)
                self._model[label].append(normalized)

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
        signature = self._signature(item)
        normalized = self._normalize(signature)

        min_distance = -1
        matched_label = ''

        for label, images in self._model.items():
            for image in images:
                distance = jellyfish.damerau_levenshtein_distance(
                    image, normalized)

                if min_distance < 0 or distance < min_distance:
                    min_distance = distance
                    matched_label = label

        return matched_label

    def _start(self, img: np.ndarray) -> Tuple[int, int, bool]:
        for col in range(img.shape[1]):
            for row in range(img.shape[0] - 1, -1, -1):
                if img[row, col] > 0:
                    return (row, col), True
        return (0, 0), False

    def _directions(self, img: np.ndarray, row: int, col: int) -> Tuple[int, List[Tuple[int, int]]]:
        # 8 1 2
        # 7 . 3
        # 6 5 3

        direction_code = 0
        directions = []

        if row > 0 and img[row-1, col] > 0:
            directions.append((row-1, col))
            direction_code = 1 if direction_code == 0 else direction_code
        if row > 0 and col < img.shape[1] - 1 and img[row-1, col+1] > 0:
            directions.append((row-1, col+1))
            direction_code = 2 if direction_code == 0 else direction_code
        if col < img.shape[1] - 1 and img[row, col+1] > 0:
            directions.append((row, col+1))
            direction_code = 3 if direction_code == 0 else direction_code
        if row < img.shape[0] - 1 and col < img.shape[1] - 1 and img[row+1, col+1] > 0:
            directions.append((row+1, col+1))
            direction_code = 4 if direction_code == 0 else direction_code
        if row < img.shape[0] - 1 and img[row + 1, col] > 0:
            directions.append((row+1, col))
            direction_code = 5 if direction_code == 0 else direction_code
        if row < img.shape[0] - 1 and col > 0 and img[row + 1, col-1] > 0:
            directions.append((row+1, col-1))
            direction_code = 6 if direction_code == 0 else direction_code
        if col > 0 and img[row, col-1] > 0:
            directions.append((row, col-1))
            direction_code = 7 if direction_code == 0 else direction_code
        if row > 0 and col > 0 and img[row-1, col-1] > 0:
            directions.append((row-1, col-1))
            direction_code = 8 if direction_code == 0 else direction_code

        return direction_code, directions

    def _signature(self, img: np.ndarray) -> Signature:
        img = img.copy()
        sign = Signature()

        while img.any():
            start, ok = self._start(img)
            if not ok:
                break

            row, col = start
            if img[row, col] > 0:
                s = self._signature_recurse(img, start)
                sign.add(s)

        return sign

    def _signature_recurse(self, img: np.ndarray, start: Tuple[int, int]) -> Signature:
        row, col = start
        img[row, col] = 0

        code, directions = self._directions(img, row, col)
        sign = Signature(code)

        for row, col in directions:
            if img[row, col] > 0:
                alt_sign = self._signature_recurse(img, (row, col))
                sign.add(alt_sign)

        return sign

    def _normalize(self, signature: Signature) -> List[int]:
        signature = str(signature)

        patterns = {
            '(0)': '',
            '0': '',
        }

        for i in range(1, 9):
            key = str(i) + str(i)
            patterns[key] = str(i)

        for i in range(2, 9, 2):
            for offset in [-1, 1]:
                key = str(i + offset) + str(i) + str(i + offset)
                patterns[key] = str(i)

                key = str(i) + str(i + offset) + str(i)
                patterns[key] = str(i)

        r = '(.*)\(([0-9]+)\(([0-9]+)\)\)(.*)'

        previous = ''
        while previous != signature:
            previous = signature

            for old, new in patterns.items():
                signature = signature.replace(old, new)

            match = re.match(r, signature)
            if match:
                signature = f'{match.group(1)}({match.group(2)}{match.group(3)}){match.group(4)}'

        return signature
