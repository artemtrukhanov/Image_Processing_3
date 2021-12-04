import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import glob


def DFFTnp(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    return fshift


def reverseDFFTnp(dfft):
    f_ishift = np.fft.ifftshift(dfft)
    reverse_image = np.fft.ifft2(f_ishift)
    return reverse_image


def Gaussian(img, fshift):
    ksize = 21
    kernel = np.zeros(img.shape)
    blur = cv.getGaussianKernel(ksize, -1)
    blur = np.matmul(blur, np.transpose(blur))
    kernel[0:ksize, 0:ksize] = blur
    fkshift = DFFTnp(kernel)
    mult = np.multiply(fshift, fkshift)
    reverse_image = reverseDFFTnp(mult)
    return reverse_image


folder_path = r".\stripes/"
images = glob.glob(folder_path + '00_18.png')
for image in images:
    img = np.float32(cv.imread(image, 0))
    fshift = DFFTnp(img)

    plt.subplot(221), plt.title('Input spectrum'), plt.xticks([]), plt.yticks([])
    plt.imshow(np.abs(fshift), cmap="gray", norm=LogNorm(vmin=5))

    w, h = fshift.shape
    maxpix = fshift[w // 2][h // 2]
    for i in range(w):
        for j in range(h):
            if i != w // 2 and j != h // 2:
                if abs(np.abs(fshift[i][j]) - np.abs(maxpix)) < np.abs(maxpix) - 250000:
                    fshift[i][j] = 0

    plt.subplot(222), plt.title('Notch filter'), plt.xticks([]), plt.yticks([])
    plt.imshow(np.abs(fshift), cmap="gray", norm=LogNorm(vmin=5))

    reverse_image = Gaussian(reverseDFFTnp(fshift), fshift)

    plt.subplot(223), plt.title('Input image'), plt.xticks([]), plt.yticks([])
    plt.imshow(abs(img), cmap='gray')
    plt.subplot(224), plt.title('Gaussian result'), plt.xticks([]), plt.yticks([])
    plt.imshow(abs(reverse_image), cmap='gray')

    plt.show()