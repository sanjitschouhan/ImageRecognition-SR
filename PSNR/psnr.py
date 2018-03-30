# import the necessary packages
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import structural_similarity as ssim


def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


def compare_images(imageA, imageB, title):
    # compute the mean squared error and structural similarity
    # index for the images
    m = mse(imageA, imageB)
    s = ssim(imageA, imageB)

    # setup the figure
    fig = plt.figure(title)
    plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))

    # show first image
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(imageA, cmap=plt.cm.gray)
    plt.axis("off")

    # show the second image
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(imageB, cmap=plt.cm.gray)
    plt.axis("off")

    # show the images
    # plt.show()


original = cv2.imread("Original.png")
low = cv2.imread("HALF.png")
low = cv2.resize(low, (256, 256), interpolation = cv2.INTER_CUBIC)
sr = cv2.imread("HALF_scaled(2x).png")
sr = cv2.resize(sr, (256, 256), interpolation = cv2.INTER_CUBIC)

# convert the images to grayscale
original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
low = cv2.cvtColor(low, cv2.COLOR_BGR2GRAY)
sr = cv2.cvtColor(sr, cv2.COLOR_BGR2GRAY)

# initialize the figure
fig = plt.figure("Images")
images = ("Original", original), ("LOW", low), ("SR", sr)

# loop over the images
for (i, (name, image)) in enumerate(images):
    # show the image
    ax = fig.add_subplot(1, 3, i + 1)
    ax.set_title(name)
    plt.imshow(image, cmap=plt.cm.gray)
    plt.axis("off")

# show the figure
plt.show()

# compare the images
# compare_images(original, original, "Original vs. Original")
compare_images(original, low, "Original vs. LOW")
compare_images(original, sr, "Original vs. SR")

plt.show()
