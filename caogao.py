import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
import glob

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:] # mask.shape = (3, 1200, 1800), mask.shape[-2:] = (1200, 1800)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1) # shape (1200, 1800, 1) x shape (1, 1, 4) = shape (1200, 1800, 4)
    ax.imshow(mask_image) # plt.gca().imshow(mask_image) plots the 1s and 0s, allowing visualization

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

# arr = np.arange(100).reshape((10, 10))
# fig = plt.figure(figsize =(4, 4))

# h, w = mask.shape[-2:] <-- what does [-2:] do?
def what_this_array_manipulation():
    arr = np.array([[1,2,3], [4,5,6], [7,8,9]])
    print(arr.shape) # (3, 3)
    print(arr)
    # [[1 2 3]
    #  [4 5 6]
    #  [7 8 9]]
    arr= arr[-2:] # remains the last 2 dimension
    print(arr.shape) # (2, 3)
    print(arr)
    # [[4 5 6]
    #  [7 8 9]]
    arr= arr[-1:] # remains the last 1 dimension
    print(arr.shape) # (2, 3)
    print(arr)
# what_this_array_manipulation()

def array_multiplication_shape():
    color = np.array([1, 2, 3, 0.6])
    print('np.array([1, 2, 3, 0.6]) shape: ', color.shape)
    print(color)
    # reshape an array into a 3-dimensional array where the first and second dimensions have a length of 1 each, 
    # and the third dimension is determined automatically to fit the total number of elements in the original array.
    color = color.reshape(1, 1, -1)
    print('color.reshape(1, 1, -1) shape: ', color.shape) # (1, 1, 4)
    print(color, '\n')
    arr = np.array([[1,2,3], [4,5,6], [7,8,9]])
    h, w = arr.shape[-2:]
    mask_image = arr.reshape(h, w, 1)
    print('arr.reshape(h, w, 1) shape: ',mask_image.shape) # (3, 3, 1)
    print(mask_image , '\n')
    mask_image = arr.reshape(h, w, 1) * color.reshape(1, 1, -1)
    print('arr.reshape(h, w, 1) * color.reshape(1, 1, -1) shape: ', mask_image.shape) # (3, 3, 4)
    print(mask_image)
# array_multiplication_shape()

def what_is_plt_gca():
    # create a numpy array of zeros with dimension of 500 x 500, 
    # then makes the middle 100 rows to  be ones
    array_shape = (500, 500)
    zeros_array = np.zeros(array_shape)
    middle_start = (array_shape[0] - 100) // 2
    middle_end = middle_start + 100
    zeros_array[middle_start:middle_end, :] = 1
    plt.gca().imshow(zeros_array)
    plt.show()
# what_is_plt_gca()

# image = cv2.imread('images/truck.jpg')

images = [cv2.imread(image) for image in glob.glob("images/bmx-trees_frame/*.jpg")]

image = images[0]

cv2.namedWindow("Get_mask", cv2.WND_PROP_FULLSCREEN)
x, y, w, h = cv2.selectROI("Get_mask", image, showCrosshair=False, fromCenter=False)
input_box = np.array([x, y, x+w, y+h])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image)
# show_box(input_box, plt.gca())
plt.axis('off')
plt.show()