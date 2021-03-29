import preprocessing

images = preprocessing.load_images('./data/images')
images = preprocessing.process(images, (32, 32))
preprocessing.save_images('./data/images-gen', images)
