import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def show_box(box, image):
    x_start, y_start = box[0], box[1]
    x_end, y_end = box[2], box[3]
    cv2.rectangle(image, (x_start, y_start), (x_end, y_end), color=(0,255,0), thickness=2)

# take a mask, return input_box numpy array
def create_box_from_mask(mask):
    true_indices = np.argwhere(mask)

    # Calculate bounding box coordinates
    min_x = true_indices[:, 1].min() # x0
    max_x = true_indices[:, 1].max()
    min_y = true_indices[:, 0].min() # y0
    max_y = true_indices[:, 0].max()

    # Create a rectangle patch for the bounding box
    box_width = max_x - min_x
    box_height = max_y - min_y
    
    box = np.array([min_x-10, min_y-10, min_x + box_width+10, min_y + box_height+10])
    return box

def show_masked_image(masks, input_box, image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.cla()
    plt.imshow(image)
    color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = masks[0].shape[-2:]
    mask_image = masks[0].reshape(h, w, 1) * color.reshape(1, 1, -1)
    plt.gca().imshow(mask_image)

    if input_box is not None:
        box = input_box[0]
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        plt.gca().add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
    
    plt.axis('off')
    plt.pause(0.0001)

def save_black_white_mask(i, masks, data):
    mask = (masks[0] * 255)

    mask_save_path = './' + data + '_mask/'
    if not os.path.exists(mask_save_path):
        os.makedirs(mask_save_path)

    cv2.imwrite(mask_save_path + '{:05d}.png'.format(i), mask)

def create_images_from_video(data):
    video_name_with_extension = data.split('/')[-1]
    video_name_wihout_extension = video_name_with_extension.split('.')[0]
    
    save_image_path = './data/' + video_name_wihout_extension

    if not os.path.exists(save_image_path):
        os.makedirs(save_image_path)

    cam = cv2.VideoCapture(data)
    currentframe = 0
    while(True):
        ret,frame = cam.read()
        if ret:
            cv2.imwrite(save_image_path + '/{:05d}.jpg'.format(currentframe), frame)
            currentframe += 1
        else:
            break
    cam.release()
