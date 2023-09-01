import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
import glob
import matplotlib.patches as patches

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:] # mask.shape = (3, 1200, 1800), mask.shape[-2:] = (1200, 1800)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1) # shape (1200, 1800, 1) x shape (1, 1, 4) = shape (1200, 1800, 4)
    ax.imshow(mask_image) # plt.gca().imshow(mask_image) plots the 1s and 0s, allowing visualization

def show_box(box, image):
    x_start, y_start = box[0], box[1]
    x_end, y_end = box[2], box[3]
    return cv2.rectangle(image, (x_start, y_start), (x_end, y_end), color=(0,255,0), thickness=2)

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
    # plt.show()
    ones_indices = np.argwhere(zeros_array == 1)

    # Calculate bounding box coordinates
    min_x = ones_indices[:, 1].min()
    max_x = ones_indices[:, 1].max()
    min_y = ones_indices[:, 0].min()
    max_y = ones_indices[:, 0].max()

    # Create a rectangle patch for the bounding box
    bbox_width = max_x - min_x + 1
    bbox_height = max_y - min_y + 1
    bbox = patches.Rectangle((min_x - 0.5, min_y - 0.5), bbox_width, bbox_height, linewidth=2, edgecolor='r', facecolor='none')

    # Add the bounding box to the plot
    plt.gca().add_patch(bbox)

    plt.show()
# what_is_plt_gca()

def try_selectROI():
    # image = cv2.imread('images/truck.jpg')
    images = [cv2.imread(image) for image in glob.glob("images/bmx-trees/*.jpg")]
    image = images[0]
    cv2.namedWindow("Get_mask", cv2.WINDOW_NORMAL)
    x, y, w, h = cv2.selectROI("Get_mask", image, showCrosshair=False, fromCenter=False)
    input_box = np.array([x, y, x+w, y+h])
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image * np.zeros(image.shape)
    image = show_box(input_box, image)
    cv2.imshow('', image)
    cv2.imshow('', images[60])
    cv2.waitKey(0)
# try_selectROI()

def try_mask():
    images = [cv2.imread(image) for image in glob.glob("images/bmx-trees/*.jpg")]
    image = images[0]
    # create a numpy array of zeros with dimension of 500 x 500, 
    # then makes the middle 100 rows to  be ones
    array_shape = (1080, 1920)
    mask = np.zeros(array_shape, dtype=bool)
    middle_start = (array_shape[0] - 100) // 2
    middle_end = middle_start + 200
    mask[middle_start:middle_end, :] = True

    # mask = ~mask
    # mask = mask.astype(np.uint8) * 1
    # h, w = mask.shape[-2:]
    # mask = mask.reshape(h, w, 1)
    # cv2.imshow('', mask)
    # cv2.waitKey(0)
    color = np.array([255/255, 255/255, 255/255, 1])
    print(color.dtype)

    h, w = mask.shape[-2:]
    mask = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    print(mask.dtype)

    cv2.imshow('', mask)
    cv2.waitKey(0)
    cv2.imwrite('C:/Users/Kainian/Desktop/WorkSpace/IM_Ghost_Project/images/annotation/mask.jpg', mask)
    # mask_image = image * mask
    # print(mask_image.shape)


    # cv2.imshow('', mask_image)
    # cv2.waitKey(0)
# try_mask()

def draw_rectangle_from_points():
    images = [cv2.imread(image) for image in glob.glob("images/bmx-trees/*.jpg")]
    image = images[-1]

    location = np.array([559.13275, 703.2408, 573.11835, 370.38312, 771.6838, 378.72626, 757.6982, 711.5839])
    # temp_location = np.intp(location).reshape((-1, 1, 2))
    temp_location = np.reshape(np.intp(location), (4, -1))
    max_point = np.max(temp_location, axis=0)
    min_point = np.min(temp_location, axis=0)
    print(max_point,min_point)
    cv2.rectangle(image, min_point, max_point, color=(0,255,0), thickness=2)

    cv2.imshow('', image)
    cv2.waitKey(0)
draw_rectangle_from_points()

