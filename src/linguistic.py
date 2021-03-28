from typing import Dict, List, Tuple
import numpy as np

class Group:
    def __init__(self) -> None:
        self._items = []

    def add(self, code: int):
        self._items.append(code)

    def __str__(self) -> str:
        sequence = ''.join(map(str, self._items))
        return f'({sequence})'

class Signature:
    def __init__(self) -> None:
        self._sequences = []

    def add(self, group: Group):
        self._sequences.append(group)

    def __str__(self) -> str:
        return ''.join(map(str, self._sequences))


class Linguistic:
    model: Dict[str, Signature]

    def __init__(self) -> None:
        self.model = {}

    def fit(self, data: Dict[str, List[np.ndarray]]):
        for label, images in data.items():
            for image in images:
                sign = self._signature(image)
                print(sign)

    def predict(self, data: List[np.ndarray]):
        pass

    def save_model(self, path: str):
        pass

    def load_model(self, path: str):
        pass

    def _start(self, img: np.ndarray) -> Tuple[int, int, bool]:
        for row in range(img.shape[0] - 1, -1, -1):
            for col in range(img.shape[1]):
                if img[row, col] > 0:
                    return (row, col), True
        return (0, 0), False


    def _directions(self, img: np.ndarray, row: int, col: int) -> Tuple[int, List[Tuple[int, int]], bool]:
        # 8 1 2
        # 7 . 3
        # 6 5 3

        direction_code = 9
        directions = []

        if row > 0 and img[row-1, col] > 0:
            directions.append((row-1, col))
            direction_code = min(direction_code, 1)
        if row > 0 and col < img.shape[1] - 1 and img[row-1, col+1] > 0:
            directions.append((row-1, col+1))
            direction_code = min(direction_code, 2)
        if col < img.shape[1] - 1 and img[row, col+1] > 0:
            directions.append((row, col+1))
            direction_code = min(direction_code, 3)
        if row < img.shape[0] - 1 and col < img.shape[1] - 1 and img[row+1, col+1] > 0:
            directions.append((row+1, col+1))
            direction_code = min(direction_code, 4)
        if row < img.shape[0] - 1 and img[row + 1, col] > 0:
            directions.append((row+1, col))
            direction_code = min(direction_code, 5)
        if row < img.shape[0] - 1 and col > 0 and img[row + 1, col-1] > 0:
            directions.append((row+1, col-1))
            direction_code = min(direction_code, 6)
        if col > 0 and img[row, col-1] > 0:
            directions.append((row, col-1))
            direction_code = min(direction_code, 7)
        if row > 0 and col > 0 and img[row-1, col-1] > 0:
            directions.append((row-1, col-1))
            direction_code = min(direction_code, 8)

        return direction_code, directions, direction_code != 9


    def _signature(self, img: np.ndarray) -> Signature:
        img = img.copy()
        sign = Signature()

        while img.any():
            start, ok = self._start(img)
            if not ok:
                break

            paths = [start]
            while len(paths) > 0:
                row, col = paths.pop(0)
                if img[row, col] == 0:
                    continue

                directions = [(row, col)]
                group = Group()

                while len(directions) > 0:
                    row, col = directions.pop(0)
                    img[row, col] = 0

                    code, directions, ok = self._directions(img, row, col)
                    if not ok:
                        break

                    paths.extend(directions[1:])
                    group.add(code)

                sign.add(group)

        return sign

    def _normalize(img: List[int]) -> List[int]:
        pass
