import potentials
import numpy as np
import preprocessing
import prettytable

preprocessing.process('./data/images', './data/images-gen', (8, 8))

images = preprocessing.load_images('./data/images-gen', nesting=2)

pot = potentials.Potentials()
pot.fit(images)

pot.save_model('./models/potentials.json')

pot = potentials.Potentials()
pot.load_model('./models/potentials.json')

actual = []
unlabelled = []
for label, images in images.items():
    actual.extend([label for _ in images])
    unlabelled.extend(images)

predicted = pot.predict(unlabelled)

table = prettytable.PrettyTable()
table.add_column("Actual", actual)
table.add_column("Predicted", predicted)
print(table)

correctness = [x == y for x, y in zip(actual, predicted)]

total = len(actual)
correct = correctness.count(True)

print(f'Recognizing {total} images: {correct} were recognized correctly ({correct / total * 100}%)')
