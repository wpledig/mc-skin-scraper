from PIL import Image
import os

base_dir = "output/"
file_dir = {(64, 64): "full/", (64, 32): "half/"}

imgs = os.scandir(base_dir)
for img in imgs:
    im_parsed = Image.open(base_dir + img.name)
    cur_dims = im_parsed.size
    im_parsed.close()
    os.rename(base_dir + img.name, file_dir[cur_dims] + img.name)


