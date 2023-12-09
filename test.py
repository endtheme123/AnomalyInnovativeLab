import os
import numpy as np
import itertools
path = "F:/UBC-OCEAN/train_images/CC"
img_fol_dir = list(os.path.join(path, img_fol) for img_fol in os.listdir(path))
img_path = list(
                            np.random.choice(
                                list(itertools.chain.from_iterable([[os.path.join(img_fol, img)
                            for img in os.listdir(img_fol)  
                            if (os.path.isfile(os.path.join(img_fol, img))
                            and img.endswith('.png'))] for img_fol in img_fol_dir])),
                            size=512)
                            )

# print(len(img_path))
parent = os.path.abspath(os.path.join(img_path[1], "../../.."))
# print(img_path[1])
# print(parent)


global_files = [os.path.join(os.path.abspath(os.path.join(s, "../../..")), "train_thumbnails", os.path.basename(s).split("_")[0]+"_thumbnail.png") for s in img_path]
print(global_files)




