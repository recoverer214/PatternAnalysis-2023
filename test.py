PATH = '/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices'
IMAGE_SIZE = 32

import dataset as dst
import numpy as np
from collections import Counter

"""
train_loader, train_set = dst.keras_dataloader(IMAGE_SIZE, '_train/')
validation_loader, validation_set = dst.keras_dataloader(IMAGE_SIZE, '_validate/')
test_loader, test_set = dst.keras_dataloader(IMAGE_SIZE, '_test/')

print(train_set[0])



"""


b = dst.load_data_2D([PATH + '_seg_train/seg_019_week_5_slice_36.nii.gz'])
b = b.flatten()
b = Counter(b)
np.set_printoptions(threshold=np.inf)
for key, value in b.items():
    print(key, value)
