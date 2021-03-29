from shutil import Error
import potentials
import preprocessing
import prettytable


data_folder = './data/images'
train_folder = './data/train'
test_folder = './data/val'
model_folder = './models/potentials.json'

dimensions = (8, 8)


def prepare():
    images = preprocessing.load_images(data_folder)
    train, test = preprocessing.split(images, 0.8)
    train = preprocessing.process_dataset(train, dimensions)

    preprocessing.save_images(train_folder, train)
    preprocessing.save_images(test_folder, test)


def train():
    pot = potentials.Potentials()

    images = preprocessing.load_merged_images(train_folder)
    pot.fit(images)

    pot.save_model(model_folder)


def test():
    pot = potentials.Potentials()
    pot.load_model(model_folder)

    images = preprocessing.load_merged_images(test_folder)
    images = preprocessing.process_dataset(images, dimensions)

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

    print(
        f'Recognizing {total} images: {correct} were recognized correctly ({correct / total * 100}%)')


prepare()
train()
test()
