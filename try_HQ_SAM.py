import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import glob
import matplotlib.patches as patches

from segment_anything_hq import sam_model_registry, SamPredictor

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_white_mask(mask, ax):
    # R (Red) channel ranges from 0 to 1, where 0 is no red and 1 is full red intensity.
    # G (Green) channel ranges from 0 to 1, where 0 is no green and 1 is full green intensity.
    # B (Blue) channel ranges from 0 to 1, where 0 is no blue and 1 is full blue intensity.
    # A (Alpha) channel ranges from 0 to 1, where 0 is fully transparent (invisible) and 1 is fully opaque (completely visible).
    color = np.array([255/255, 255/255, 255/255, 1]) # [R, G, B, alpha] = [1,1,1,1] is white color with completely visible
    h, w = mask.shape[-2:] # mask.shape = (3, 1200, 1800), mask.shape[-2:] = (1200, 1800)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1) # shape (1200, 1800, 1) x shape (1, 1, 4) = shape (1200, 1800, 4)
    ax.imshow(mask_image) # plt.gca().imshow(mask_image) plots the 1s and 0s, allowing visualization

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

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
    
    box = np.array([[min_x-10, min_y-10, min_x + box_width + 20, min_y + box_height + 20]])
    return box

def show_res(masks, scores, input_point, input_label, input_box, image):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.cla()
        plt.cla()
        plt.figure(figsize=(10,10))
        image = image * np.zeros(image.shape) # make image black
        plt.imshow(image)
        # show_mask(mask, plt.gca())

        show_white_mask(mask, plt.gca())
        if input_box is not None:
            box = input_box[i]
            show_box(box, plt.gca())

        # if (input_point is not None) and (input_label is not None):
        #     show_points(input_point, input_label, plt.gca())
        # print(f"Score: {score:.3f}")
        plt.axis('off')
        plt.pause(2)

def show_res_multi(masks, scores, input_point, input_label, input_box, image):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for mask in masks:
        show_mask(mask, plt.gca(), random_color=True)
    for box in input_box:
        show_box(box, plt.gca())
    for score in scores:
        print(f"Score: {score:.3f}")
    plt.axis('off')
    plt.show()

# ------------------------------run SAM demo------------------------------------
sam_checkpoint = "C:/Users/Kainian/Desktop/WorkSpace/sam-hq/segment_anything/sam_hq_vit_h.pth"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

# image = cv2.imread('images/bike.jpg')
images = [cv2.imread(image) for image in glob.glob("images/bmx-trees/*.jpg")]

# input_box = np.array([[242,305,1866,840]])
cv2.namedWindow("Get_mask", cv2.WINDOW_NORMAL)
x, y, w, h = cv2.selectROI("Get_mask", images[0], showCrosshair=False, fromCenter=False)
cv2.destroyAllWindows()
input_box = np.array([[x, y, x+w, y+h]])

for i, image in enumerate(images):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)

    input_point, input_label = None, None
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        box = input_box,
        multimask_output=False,
        hq_token_only= False,
    )

    input_box = create_box_from_mask(masks[0])

    show_res(masks,scores,input_point, input_label, input_box, image)

    if i == 2:
        break
