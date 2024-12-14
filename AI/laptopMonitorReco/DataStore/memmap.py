import os
import numpy as np
from PIL import Image

def save_images_to_memmap(px=64, memmap_filename='images_memmap.dat', labels_filename='labels.npy'):
    images = []
    labels = []
    valid_extensions = (".jpg", ".jpeg", ".png")
    
    for label in ["1", "0"]:
        folder = label
        for filename in os.listdir(folder):
            if filename.lower().endswith(valid_extensions):
                img_path = os.path.join(folder, filename)
                img = Image.open(img_path).resize((px, px)).convert('RGB')
                img_array = np.array(img, dtype=np.uint8)
                images.append(img_array)
                labels.append(1 if label == "1" else 0)
    
    images = np.array(images)
    labels = np.array(labels)
    
    memmap_shape = images.shape
    memmap = np.memmap(memmap_filename, dtype='uint8', mode='w+', shape=memmap_shape)
    memmap[:] = images[:]
    memmap.flush()
    
    np.save(labels_filename, labels)


def load_data_from_memmap(memmap_filename=r'C:\Users\hp\OneDrive\Documents\AI\laptopMonitorReco\DataStore\images_memmap.dat', labels_filename=r'C:\Users\hp\OneDrive\Documents\AI\laptopMonitorReco\DataStore\labels.npy'):
    labels = np.load(labels_filename)
    num_images = labels.shape[0]
    dummy_image = Image.open(os.path.join("1", os.listdir("1")[0])).resize((64, 64)).convert('RGB')
    dummy_array = np.array(dummy_image, dtype=np.uint8)
    image_shape = dummy_array.shape
    memmap_shape = (num_images, *image_shape)
    memmap = np.memmap(memmap_filename, dtype='uint8', mode='r', shape=memmap_shape)
    images = memmap.reshape(num_images, -1).T / 255.0
    labels = labels.reshape(1, num_images)
    return images, labels


