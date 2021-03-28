import linguistic
import numpy as np
import preprocessing
import prettytable

preprocessing.process('./data/images', './data/images-gen', (8, 8))

images = preprocessing.load_images('./data/images-gen', nesting=2)

ling = linguistic.Linguistic()
ling.fit(images)

content = ling.save_model('1.txt')

ling = linguistic.Linguistic()
ling.load_model('1.txt')

actual = []
unlabelled = []
for label, images in images.items():
    actual.extend([label for _ in images])
    unlabelled.extend(images)

predicted = ling.predict(unlabelled)

table = prettytable.PrettyTable()
table.add_column("Actual", actual)
table.add_column("Predicted", predicted)
print(table)
