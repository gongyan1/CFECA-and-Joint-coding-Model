from albumentations import RandomBrightness,RandomContrast
import os
import cv2
import numpy as np

def aug_brightness(data_path, saveto):
    for a,b,c in os.walk(data_path):
        for f in c:
            print('processing ..',f)
            img = cv2.imread(os.path.join(a,f))
            if np.random.randint(0,1):
                aug = RandomBrightness(limit=0.5, p=1)
                img = aug(image=img)['image']
            else:
                img = np.zeros(img.shape)
            cv2.imwrite(os.path.join(saveto,f),img)
    
if __name__=='__main__':
    path = './train_image_2_lane_backup/'
    aug_brightness(data_path=path, saveto='./train_image_2_lane/')
