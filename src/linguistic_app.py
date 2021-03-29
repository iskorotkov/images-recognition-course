from shutil import Error
import linguistic
import potentials
import preprocessing
import prettytable


data_folder = './data/images'
train_folder = './data/train'
test_folder = './data/test'
model_folder = './models/linguistic.json'

dimensions = (8, 8)


def prepare():
    images = preprocessing.load_images(data_folder)
    train, test = preprocessing.split(images, 0.8)
    train = preprocessing.process(train, dimensions)

    preprocessing.save_images(train_folder, train)
    preprocessing.save_images(test_folder, test)


def train():
    ling = linguistic.Linguistic()

    images = preprocessing.load_merged_images(train_folder)
    ling.fit(images)

    ling.save_model(model_folder)


def test():
    ling = linguistic.Linguistic()
    ling.load_model(model_folder)

    images = preprocessing.load_merged_images(test_folder)
    images = preprocessing.process(images, dimensions)

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

    correctness = [x == y for x, y in zip(actual, predicted)]

    total = len(actual)
    correct = correctness.count(True)

    print(
        f'Recognizing {total} images: {correct} were recognized correctly ({correct / total * 100}%)')


prepare()
train()
test()
