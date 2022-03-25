'''Run this file once. Creates dataset that is faster to load'''

import os
import numpy as np
from skimage import io
from skimage.transform import resize

def resize_images(dir):
    classes = [name for name in os.listdir(dir)]

    for cl_idx, cl in enumerate(classes):
        parent_path = dir + "\\" + cl
        image_paths = [parent_path+"\\"+name for name in os.listdir(parent_path)]

        for path in image_paths:
            resized_image = resize(io.imread(path), (112, 112), anti_aliasing=True).astype(np.uint8)
            os.remove(path)
            io.imsave(path, resized_image)
        print("Resized Class (", cl_idx, "): ", cl)

if __name__ == "__main__":
    dir = "../data/kvasir-dataset-v2-cropped"
    resize_images(dir)