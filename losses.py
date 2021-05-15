import numpy as np
import torch
import torchvision
import torchvision.transforms as TF
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import cv2

def pyr_downsample(x):
    return x[:, :, ::2, ::2]


def pyr_upsample(x, kernel, op0, op1):
    n_channels, _, kw, kh = kernel.shape
    return F.conv_transpose2d(x, kernel, groups=n_channels, stride=2, padding=2, output_padding=(op0, op1))

def gauss_kernel5(channels=3, cuda=True):
    kernel = torch.FloatTensor([[1., 4., 6., 4., 1],
                           [4., 16., 24., 16., 4.],
                           [6., 24., 36., 24., 6.],
                           [4., 16., 24., 16., 4.],
                           [1., 4., 6., 4., 1.]])
    kernel /= 256.
    kernel = kernel.repeat(channels, 1, 1, 1)
    # print(kernel)
    if cuda:
        kernel = kernel.cuda()
    return Variable(kernel, requires_grad=False)


def build_gauss_kernel(size=5, sigma=1.0, n_channels=1, cuda=False):
    if size % 2 != 1:
        raise ValueError("kernel size must be uneven")
    grid = np.float32(np.mgrid[0:size, 0:size].T)
    gaussian = lambda x: np.exp((x - size // 2) ** 2 / (-2 * sigma ** 2)) ** 2
    kernel = np.sum(gaussian(grid), axis=2)
    kernel /= np.sum(kernel)
    # repeat same kernel across depth dimension
    kernel = np.tile(kernel, (n_channels, 1, 1))
    # conv weight should be (out_channels, groups/in_channels, h, w),
    # and since we have depth-separable convolution we want the groups dimension to be 1
    kernel = torch.FloatTensor(kernel[:, None, :, :])
    # kernel = gauss_kernel5(n_channels)
    if cuda:
        kernel = kernel.cuda()
    return Variable(kernel, requires_grad=False)


def conv_gauss(img, kernel):
    """ convolve img with a gaussian kernel that has been built with build_gauss_kernel """
    n_channels, _, kw, kh = kernel.shape
    img = F.pad(img, (kw // 2, kh // 2, kw // 2, kh // 2), mode='replicate')
    return F.conv2d(img, kernel, groups=n_channels)


def laplacian_pyramid(img, kernel, max_levels=5):
    current = img
    pyr = []

    for level in range(max_levels-1):
        filtered = conv_gauss(current, kernel)
        diff = current - filtered
        pyr.append(diff)
        current = F.avg_pool2d(filtered, 2)

    pyr.append(current) # high -> low
    return pyr

def laplacian_pyramid_expand(img, kernel, max_levels=5):
    current = img
    pyr = []
    for level in range(max_levels):
        # print("level: ", level)
        filtered = conv_gauss(current, kernel)
        down = pyr_downsample(filtered)
        up = pyr_upsample(down, 4*kernel, 1-filtered.size(2)%2, 1-filtered.size(3)%2)

        diff = current - up
        pyr.append(diff)

        current = down
    return pyr


class LapLoss(nn.Module):
    def __init__(self, max_levels=5):
        super(LapLoss, self).__init__()
        self.max_levels = max_levels
        self._gauss_kernel = None

    def forward(self, input, target):
        if self._gauss_kernel is None or self._gauss_kernel.shape[1] != input.shape[1]:
            self._gauss_kernel = gauss_kernel5(input.shape[1], cuda=input.is_cuda)

        pyr_input = laplacian_pyramid_expand(input, self._gauss_kernel, self.max_levels)
        pyr_target = laplacian_pyramid_expand(target, self._gauss_kernel, self.max_levels)
        weights = [1, 2, 4, 8, 16]

        # return sum(F.l1_loss(a, b) for a, b in zip(pyr_input, pyr_target))
        return sum(weights[i] * F.l1_loss(a, b) for i, (a, b) in enumerate(zip(pyr_input, pyr_target))).mean() 
        

def cv_laplaian_demo(pyramid_images, src):
    level = len(pyramid_images)
    for i in range(level-1, -1, -1):
        if (i-1) < 0:
            h, w = src.shape[:2]
            expand = cv2.pyrUp(pyramid_images[i], dstsize=(w, h))
            lpls = np.float32(src) - np.float32(expand) + 127
            cv2.imshow("cv_lap", np.clip(lpls, 0, 255).astype('uint8'))
            cv2.imwrite("cv_%d.jpg" % i, np.clip(lpls, 0, 255).astype('uint8'))
        else:
            h, w = pyramid_images[i-1].shape[:2]
            expand = cv2.pyrUp(pyramid_images[i], dstsize=(w, h))
            lpls = np.float32(pyramid_images[i-1]) - np.float32(expand) + 127
            cv2.imshow("cv_lap", np.clip(lpls, 0, 255).astype('uint8'))
            cv2.imwrite("cv_%d.jpg" % i, np.clip(lpls, 0, 255).astype('uint8'))
        cv2.waitKey(0)

def cv_pyramid_up(image, level=3):
    temp = image.copy()
    # cv.imshow("input", image)
    pyramid_images = []
    for i in range(level):
        dst = cv2.pyrDown(temp)
        pyramid_images.append(dst)
        # cv.imshow("pyramid_up_" + str(i), dst)
        temp = dst.copy()
    return pyramid_images

if __name__ == '__main__':
    # build_gauss_kernel()
    img = cv2.imread("00000000.png")
    img = np.float32(img).transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)

    kernel = gauss_kernel5(channels=3, cuda=False)

    pyramids = laplacian_pyramid_expand(img, kernel)
    for i, pyr in enumerate(pyramids):
        test = pyr[0].numpy().transpose(1, 2, 0)
        test = np.clip(test + 127, 0, 255).astype('uint8')
        cv2.imshow("lap_pyr_diff", test)
        cv2.imwrite("torch_%d.jpg" % i, test)
        cv2.waitKey(0)
    # pyramids = laplacian_pyramid(img, kernel)
    # test(torch.ones(size=(1, 1, 5, 5)))

    img = cv2.imread("00000000.png")
    cv_prys = cv_pyramid_up(img, 5)
    cv_laplaian_demo(cv_prys, img)
