from cv2 import cv2

img = cv2.imread('train1.jpg')

# repeat = 1000
# for r in range(repeat):
#     cv2.imwrite('repeat.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
#     if r % 100 == 0:
#         cv2.imwrite('repeat' + str(r) + '.jpg', img)
#         if r > 0:
#             cv2.imwrite('pos' + str(r) + '.bmp', pos)
#             cv2.imwrite('neg' + str(r) + '.bmp', neg)
#
#     img_2 = cv2.imread('repeat.jpg')
#
#     pos = img - img_2
#     neg = img_2 - img
#
#     img = img_2

import mxnet as mx
from PIL import Image
import numpy as np
header = mx.recordio.IRHeader(flag=0, label=0, id=0, id2=0)
img_path = 'train1.jpg'
img = mx.image.imread(img_path).asnumpy()
repeat = 100
for r in range(repeat):
    s = mx.recordio.pack_img(header, img, quality=95, img_fmt='.jpg')
    _, s = mx.recordio.unpack(s)
    img = mx.image.imdecode(s).asnumpy()

    if r % 10 == 0:
        cv2.imwrite('repeat' + str(r) + '.jpg', img)
