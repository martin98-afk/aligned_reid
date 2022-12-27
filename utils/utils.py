import cv2
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from numpy import array
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader
from torchvision import transforms

import models
from utils.datasets import MyDataset
from utils.feature_extractor import FeatureExtractor

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
np.random.seed(1)


def get_dataloader(imgs, batch_size=20):
    dataset = MyDataset(imgs, img_transform)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size)
    return dataloader


def get_feature(feature_extractor, dataloader):
    features = None
    with torch.no_grad():
        for img_batch in dataloader:
            f = feature_extractor(img_batch.to('cuda'))
            a = pool2d(f, type="max")
            if features is not None:
                features = np.vstack([features, a])
            else:
                features = a
    for i in range(features.shape[0]):
        features[i] = normalize(features[i])
    return features


def read_image(path):
    image = Image.open(path)
    image = np.array(image.resize((128, 256)))
    return image


def read_images(paths):
    images = [Image.open(path) for path in paths]
    images = np.array([np.array(image.resize((128, 256))) for image in images])
    return images


def img_to_tensor(img, tensor):
    img = torch.from_numpy(img.astype(np.float32) / 255.0)
    img = img.unsqueeze(0) if len(img.shape) < 4 else img
    img = img.permute(0, 3, 1, 2)
    return img


def _traceback(D, i, j):
    p = [i]
    q = [j]
    while (i > 0) or (j > 0):
        tb = np.argmin((D[i, j - 1], D[i - 1, j], D[i - 1, j - 1]))
        if tb == 0:
            j -= 1
        elif tb == 1:  # (tb==1)
            i -= 1
        else:
            j -= 1
            i -= 1
        p.insert(0, i)
        q.insert(0, j)
    return array(p), array(q)


def dtw(dist_mat):
    """
    no parallel version

    :param dist_mat:
    :return:
    """
    m, n = dist_mat.shape[:2]
    dist = np.zeros((m + 1, n + 1)).astype(np.float16)
    for i in range(1, m + 1):
        dist[0, i] = dist[0, i - 1] + dist_mat[0, i - 1]
    for j in range(1, n + 1):
        dist[j, 0] = dist[j - 1, 0] + dist_mat[j - 1, 0]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dist[i, j] = \
                np.min(np.stack(
                    [dist[i - 1, j], dist[i, j - 1], dist[i - 1, j - 1]],
                    axis=0), axis=0) + dist_mat[i - 1, j - 1]
    # find the shortest distance
    D = dist[1:, 1:]
    min1 = np.min(D[-1, :])
    min2 = np.min(D[:, -1])
    if min1 < min2:
        i = D.shape[0] - 1
        j = np.argmin(D[-1, :])
    else:
        j = D.shape[1] - 1
        i = np.argmin(D[:, -1])
    # find the path to get shortest distance
    path_x, path_y = _traceback(D, i, j)
    return D[i, j] / len(path_x), dist[1:, 1:], [path_x, path_y]


def batch_dtw(dist_mat):
    """
    parallel version dynamic time warping

    :param dist_mat: shape(N, M, feat, feat)
    :return:
    """
    prob_num, gallery_num, m, n = dist_mat.shape
    dist = np.zeros((prob_num, gallery_num, m + 1, n + 1)).astype(np.float16)
    for i in range(1, m + 1):
        dist[:, :, 0, i] = dist[:, :, 0, i - 1] + dist_mat[:, :, 0, i - 1]
    for j in range(1, n + 1):
        dist[:, :, j, 0] = dist[:, :, j - 1, 0] + dist_mat[:, :, j - 1, 0]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dist[:, :, i, j] = \
                np.min(np.stack(
                    [dist[:, :, i - 1, j], dist[:, :, i, j - 1], dist[:, :, i - 1, j - 1]],
                    axis=0), axis=0) + dist_mat[:, :, i - 1, j - 1]
    # find the shortest distance
    D = dist[:, :, 1:, 1:]
    return np.min(np.stack([np.min(D[:, :, -1, :], axis=-1),
                            np.min(D[:, :, :, -1], axis=-1)],
                           axis=0), axis=0) / min(m, n)


def show_alignedreid(img1, img2, dist):
    def draw_line(img, similarity):
        for i in range(1, len(similarity)):
            cv2.line(img, (0, i * 16), (63, i * 16), color=(0, 255, 0))
            cv2.line(img, (96, i * 16), (160, i * 16), color=(0, 255, 0))

    def draw_path(img, path):
        for i in range(len(path[0])):
            cv2.line(img, (64, 8 + 16 * path[0][i]), (96, 8 + 16 * path[1][i]), color=(255, 255, 0))

    img1 = cv2.resize(img1, (64, 128))
    img2 = cv2.resize(img2, (64, 128))
    img = np.zeros((128, 160, 3)).astype(img1.dtype)
    img[:, :64, :] = img1
    img[:, -64:, :] = img2
    draw_line(img, dist)
    d, D, sp = dtw(dist)
    origin_dist = (np.diag(dist)).mean()
    draw_path(img, sp)
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1).set_title(
        'Aligned distance: %.4f \n Original distance: %.4f' % (d, origin_dist))
    plt.subplot(1, 3, 1).set_xlabel('Aligned Result')
    plt.imshow(img)
    plt.subplot(1, 3, 2).set_title('Distance Map')
    plt.subplot(1, 3, 2).set_xlabel('Right Image')
    plt.subplot(1, 3, 2).set_ylabel('Left Image')
    plt.imshow(dist.astype(np.float))
    plt.subplot(1, 3, 3).set_title('dynamic time warping map')
    plt.subplot(1, 3, 3).set_xlabel('Right Image')
    plt.subplot(1, 3, 3).set_ylabel('Left Image')
    plt.imshow(D.astype(np.float))
    # draw path in mat
    for i in range(len(sp[0]) - 1):
        plt.plot(sp[1][i:i + 2], sp[0][i:i + 2], c='r')

    plt.subplots_adjust(bottom=0.1, left=0.075, right=0.85, top=0.9)
    cax = plt.axes([0.9, 0.25, 0.025, 0.5])
    plt.colorbar(cax=cax)
    plt.show()


def pool2d(tensor, type='max'):
    sz = tensor.shape
    if type == 'max':
        x = torch.nn.functional.max_pool2d(tensor, kernel_size=(1, sz[3]))
    if type == 'mean':
        x = torch.nn.functional.avg_pool2d(tensor, kernel_size=(1, sz[3]))
    x = np.squeeze(x.cpu().data.numpy().astype(np.float16))
    x = x[np.newaxis, ...] if len(x.shape) < 3 else x
    x = np.transpose(x, (0, 2, 1))
    return x


def load_model(device='cuda:0', path="./checkpoint_ep300.pth.tar"):
    model = models.init_model(name='resnet50', num_classes=751, loss={'softmax', 'metric'},
                              use_gpu=False, aligned=True)
    checkpoint = torch.load(path, encoding='latin-1')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    myexactor = FeatureExtractor(model, ['7'])
    myexactor.to(device)
    return myexactor
