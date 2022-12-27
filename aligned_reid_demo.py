import time
from glob import glob

import numpy as np

from utils.local_dist import batch_euclidean_dist
from utils.utils import read_images, load_model, batch_dtw, get_feature, get_dataloader

if __name__ == "__main__":
    image_paths = glob("Market-1501-v15.09.15/query/*.jpg")
    probe_paths = np.random.choice(image_paths, 20)
    gallery_paths = list(set(image_paths) - set(probe_paths))
    probe_imgs = read_images(probe_paths)
    gallery_imgs = read_images(gallery_paths)

    myexactor = load_model()
    # img_path1 = "Market-1501-v15.09.15/query/0001_c2s1_000301_00.jpg"
    # img_path2 = "Market-1501-v15.09.15/query/0003_c1s6_015971_00.jpg"
    #
    # img1 = read_image(img_path1)
    # img2 = read_image(img_path2)
    # img10, img20 = img1.copy(), img2.copy()
    #
    # probe_imgs = img1[np.newaxis, ...] if len(img1.shape) < 4 else img1
    # gallery_imgs = img2[np.newaxis, ...] if len(img2.shape) < 4 else img2

    probe_dataloader = get_dataloader(probe_imgs)
    gallery_dataloader = get_dataloader(gallery_imgs)
    print('------------------提取特征中-------------------')
    t = time.time()
    probe_feat = get_feature(myexactor, probe_dataloader)
    gallery_feat = get_feature(myexactor, gallery_dataloader)
    print('------------------特征提取完毕-----------------')
    print('消耗时间为：', time.time() - t)
    print('-----------------计算距离矩阵中----------------')
    t = time.time()
    # TODO 目前使用cpu计算矩阵运算过慢，后续改为使用gpu计算
    dist = batch_euclidean_dist(probe_feat, gallery_feat)
    print('消耗时间为：', time.time() - t)
    print('----------------进行dtw动态匹配中--------------')
    t = time.time()
    result = batch_dtw(dist)
    reuslt_id = np.argsort(result, axis=1)
    print('消耗时间为：', time.time() - t)
    # show_alignedreid(img10, img20, np.squeeze(dist))
