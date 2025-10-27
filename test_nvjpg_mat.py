from typing import Tuple

import cv2
import numpy as np
from nvjpeg import NvJpeg
import time
import os

nj = NvJpeg()

# mat_size = [(2160, 3840, 3), (1080, 1920, 3), (720, 1280, 3)]
mat_size = [(2160, 3840, 3)]
params = [cv2.IMWRITE_JPEG_QUALITY, 99]

np.random.seed(0) # this will make sure the generated image is the same each run

jpg_counter = 1

# while(1):
#     for i in range(len(mat_size)):
#         height, width, channels = mat_size[i]
#         # for j in range(0, 10):
#         #     noise_img = np.random.randint(0, 128, (height, width, channels), dtype=np.uint8)
#         #     cv_start = time.time()
#         #     success, noise_img_bytes = cv2.imencode('.jpg', noise_img, params)
#         #     nv_start = time.time()
#         #     compressed_bytes = nj.encode(noise_img, 99)
#         #     # print(f"cv exec time：{nv_start - cv_start} s, nvjpeg exec time: {time.time() - nv_start} s")
#         #     print(f'cv encoded size: {noise_img_bytes.nbytes/1024/1024}MB, nvjpeg encoded size: {len(compressed_bytes)/1024/1024}MB')

#         for j in range(0, 10):
#             noise_img = np.random.randint(0, 256, (height, width, channels), dtype=np.uint8)
#             cv_start = time.time()
#             success, noise_img_bytes = cv2.imencode('.jpg', noise_img, params)
#             nv_start = time.time()
#             compressed_bytes = nj.encode(noise_img, 99)
#             # print(f"cv exec time：{nv_start - cv_start} s, nvjpeg exec time: {time.time() - nv_start} s")
#             print(f'cv encoded size: {noise_img_bytes.nbytes/1024/1024}MB, nvjpeg encoded size: {len(compressed_bytes)/1024/1024}MB')

#             # fp = open(os.path.join(os.path.dirname(__file__), "out", f"{jpg_counter}.jpg"), "wb")
#             # fp.write(noise_img_bytes)
#             # fp.close()

#             # fp = open(os.path.join(os.path.dirname(__file__), "out", f"{jpg_counter}_n.jpg"), "wb")
#             # fp.write(compressed_bytes)
#             # fp.close()

#             jpg_counter = jpg_counter + 1


for i in range(len(mat_size)):
    height, width, channels = mat_size[i]
    for j in range(0, 2):
        noise_img = np.random.randint(0, 256, (height, width, channels), dtype=np.uint8)
        cv_start = time.time()
        success, noise_img_bytes = cv2.imencode('.jpg', noise_img, params)
        nv_start = time.time()
        compressed_bytes = nj.encode(noise_img, 99)
        # print(f"cv exec time：{nv_start - cv_start} s, nvjpeg exec time: {time.time() - nv_start} s")
        print(f'cv encoded size: {noise_img_bytes.nbytes/1024/1024}MB, nvjpeg encoded size: {len(compressed_bytes)/1024/1024}MB')

        fp = open(os.path.join(os.path.dirname(__file__), "out", f"{jpg_counter}.jpg"), "wb")
        fp.write(noise_img_bytes)
        fp.close()

        fp = open(os.path.join(os.path.dirname(__file__), "out", f"{jpg_counter}_n.jpg"), "wb")
        fp.write(compressed_bytes)
        fp.close()

        jpg_counter = jpg_counter + 1
