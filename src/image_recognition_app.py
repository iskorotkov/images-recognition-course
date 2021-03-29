import os
from shutil import Error

import matplotlib
import linguistic
import potentials
import preprocessing
import matplotlib.pyplot as plt

matplotlib.use('tkagg')

dimensions = (8, 8)
data = './data/test'

ling = linguistic.Linguistic()
ling.load_model('./models/linguistic.json')

pot = potentials.Potentials()
pot.load_model('./models/potentials.json')

files = []

for dirpath, dirnames, filenames in os.walk(data):
    filenames = [os.path.join(dirpath, filename) for filename in filenames]
    files.extend(filenames)

print('Enter 0 to exit')
print('Select image')

for index, file in enumerate(files):
    print(f'{index+1}) {file}')

while True:
    i = input('> ')
    try:
        num = int(i)
    except Error as e:
        print(e)
        break

    if num <= 0 or num > len(files):
        break

    filepath = files[num - 1]
    image = preprocessing.load_image(filepath)
    image = preprocessing.process_image(image, dimensions)

    ling_label = ling.predict([image])[0]
    pot_label = pot.predict([image])[0]

    plt.figure(0)
    plt.suptitle('Image')
    plt.text(0, 0, f'Linguistic method: {ling_label}', bbox=dict(
        facecolor='white', alpha=0.8))
    plt.text(0, 1, f'Potentials method: {pot_label}', bbox=dict(
        facecolor='white', alpha=0.8))
    plt.imshow(image, cmap='gray')
    plt.show()

print('Terminated')
