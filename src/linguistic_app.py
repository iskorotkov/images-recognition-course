import linguistic
import numpy as np

ling = linguistic.Linguistic()

ling.fit({'a': [
    np.array([
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1]
    ])
]})
