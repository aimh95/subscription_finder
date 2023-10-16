import cv2
import tensorflow as tf
from PIL import Image
import binascii
import numpy as np

data = []

with open("./outimage_1.gray", encoding="euc-kr") as f:
    try:
        output = []
        for line in f:
            for char in line:
                output.append(ord(char))
    except:
        print("error")
#440 160

output_np = np.zeros((160, 440))
cnt = 0
for y in range(30):
    for x in range(440):
        output_np[y, x] = output[cnt]
        cnt+=1

output_image = Image.fromarray(output_np*255)
output_image.show()




with open("./outimage_png", "wb") as f:
    f.write(data)