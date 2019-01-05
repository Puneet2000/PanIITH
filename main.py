from PIL import Image
import numpy as np
import pandas as pd

import matplotlib

import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# image_file = pd.read_csv('./data/solution.csv')
# image_ids = image_file['ids']
# num_images = len(image_ids)

# all_grayscale_images = []
# for i in range(num_images):
# 	img = Image.open('./data/train_data/' + str(image_ids[i]) + '.png').convert('L')
# 	img = img.resize((128, 128), Image.ANTIALIAS)
# 	data = np.array(img).flatten()
# 	all_grayscale_images.append(data)

# all_grayscale_images = np.array(all_grayscale_images)

# np.save('./data/train_images_numpy', all_grayscale_images)

all_grayscale_images = np.load('./data/train_images_numpy.npy')

print("Size are -> {}".format(all_grayscale_images.shape))

tsne = TSNE(n_components=2, random_state=0)
print("hi")
# 2d_visual = tsne.fit_transform(all_grayscale_images)

# 2d_visual_x = 2d_visual[:, 0]
# 2d_visual_y = 2d_visual[:, 1]

# fig, ax = plt.subplots(figsize=(10,10))
# ax.scatter(2d_visual_x, 2d_visual_y)

# plt.title("2d view of image featues ")
# plt.savefig('initial.png')
# plt.close()