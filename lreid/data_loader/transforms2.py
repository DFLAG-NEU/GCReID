import imp
import torch
import random
import math


class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) >= self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img

'''
import torchvision.transforms as transforms
from PIL import Image
from IPython import embed

trans_1 = transforms.ColorJitter(brightness=2, contrast=0.5, saturation=0.5, hue=0.2)
            # transforms.RandomHorizontalFlip(p=0.5)
trans_2 = transforms.RandomRotation(degrees=30) # 随机旋转30度
trans_3 = RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406]) 

# path1 = '/media/lzs/de2ef254-eaa4-4486-b00b-ab367ed2a6d8/home/lzs/LifelongReID_new/dataset/market1501/Market-1501-v15.09.15/query/0001_c1s1_001051_00.jpg'
# path2 = '/media/lzs/de2ef254-eaa4-4486-b00b-ab367ed2a6d8/home/lzs/LifelongReID_new/dataset/market1501/Market-1501-v15.09.15/query/0006_c3s3_075694_00.jpg'
# path3 = '/media/lzs/de2ef254-eaa4-4486-b00b-ab367ed2a6d8/home/lzs/LifelongReID_new/dataset/market1501/Market-1501-v15.09.15/query/0033_c3s3_062903_00.jpg'

path1 = '/home/lzs/Desktop/trans_2/i1.jpg'
path2 = '/home/lzs/Desktop/trans_2/i2.jpg'
path3 = '/home/lzs/Desktop/trans_2/i3.jpg'

p1 = '/home/lzs/Desktop/i1.jpg'
p2 = '/home/lzs/Desktop/i2.jpg'
p3 = '/home/lzs/Desktop/i3.jpg'

i1 = Image.open(path1)
i2 = Image.open(path2)
i3 = Image.open(path3)
embed()
ii1 = trans_3(i1)
# ii1.save(p1)
ii2 = trans_3(i2)
# ii2.save(p2)

ii3 = trans_3(i3)
'''


