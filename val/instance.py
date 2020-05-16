import os
import cv2

for a,b,c in os.walk('./lane_label/'):
    for f in c:
        print('processing .. ', f)
        path = os.path.join(a,f)
        im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        im[im>0] = 255
        cv2.imwrite(os.path.join('./lane_label_instance/',f),im)
