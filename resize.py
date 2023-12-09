import os
import numpy as np
import itertools
import cv2
path_in = "F:/UBC-OCEAN/train_thumbnails"
path_out = "F:/UBC-OCEAN/global_images"

img_path = list(
                                [os.path.join(path_in, img)
                            for img in os.listdir(path_in)
                            if (os.path.isfile(os.path.join(path_in,
                            img)) and img.endswith('png'))]
                            )
                            
for img in img_path:

    image = cv2.imread(img)
    img_out = cv2.resize(image, (224, 224))
    cv2.imwrite(os.path.join(path_out, os.path.basename(img).split("_")[0]+"_global.png")
, img_out) 