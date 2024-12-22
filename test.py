PATH = '/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices'
IMAGE_SIZE = 32

import dataset as dst
import numpy as np
from collections import Counter

test_loader, test_set = dst.keras_dataloader(IMAGE_SIZE, '_test/')

for epoch in range(NUM_EPOCHS):
        print(str(epoch + 1) + '/' + str(NUM_EPOCHS) + 'th iteration')
        model.train()
        train_loss = 0.0
        val_loss = 0.0
        for raw, seg in train_loader:
            #Batch index, raw data and segmented data
            imgs, labels = raw.to(DEVICE), seg.to(DEVICE) #(12, 1, 256, 128) (12, 6, 256, 128)
            #(Batch size, Channel number, Width, Height)
            optimizer.zero_grad()
            model.eval()
                    with torch.no_grad():
                        for raw, seg in validation_loader:
                            imgs, labels = raw.to(DEVICE), seg.to(DEVICE)
                            outputs = model(imgs)
                            loss = criteria(outputs, labels)
                            loss.backward()
                            val_loss += loss.item()

b = dst.load_data_2D([PATH + '_seg_train/seg_019_week_5_slice_36.nii.gz'])
b = b.flatten()
b = Counter(b)
np.set_printoptions(threshold=np.inf)
for key, value in b.items():
    print(key)
