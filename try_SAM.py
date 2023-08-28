import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import glob

from segment_anything import sam_model_registry, SamPredictor
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

# ------------------------------run SAM demo------------------------------------
sam_checkpoint = "C:/Users/Kainian/Desktop/WorkSpace/segment-anything/segment_anything/sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)

image = cv2.imread('images/bike_people.jpg')
# images = [cv2.imread(image) for image in glob.glob("images/bmx-trees/*.jpg")]
# image = images[0]

# input_box = np.array([64,76,940,919])
cv2.namedWindow("Get_mask", cv2.WND_PROP_FULLSCREEN)
x, y, w, h = cv2.selectROI("Get_mask", image, showCrosshair=False, fromCenter=False)
box_points = np.array([x, y, x+w, y+h])
input_box = np.array(box_points)
print(input_box)

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
predictor.set_image(image)

masks, _, _ = predictor.predict(
    point_coords=None,
    point_labels=None,
    box=input_box[None, :],
    multimask_output=False,
)

plt.figure(figsize=(10, 10))
# image = image * np.zeros(image.shape) # make image black 
plt.imshow(image)
show_mask(masks[0], plt.gca())
# show_white_mask(masks[0], plt.gca())
show_box(input_box, plt.gca())
plt.axis('off')
plt.savefig('image_sam.jpg')
plt.show()