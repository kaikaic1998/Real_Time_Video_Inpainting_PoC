import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import os
import argparse

# import Siammask modules
from Siammask.get_mask.test import *
from Siammask.get_mask.models.custom import Custom

# import Deep Video Inpainting modules
from Deep_Video_Inpainting.inpaint import inpaint

# import HQ_SAM modules
from segment_anything_hq import sam_model_registry, SamPredictor

parser = argparse.ArgumentParser(description='Demo')

parser.add_argument('--resume', default='Model_CP/SiamMask_DAVIS.pth', type=str,
                    metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--data', default='data/car-turn', help='videos or image files')
parser.add_argument('--mask-dilation', default=32, type=int, help='mask dilation when inpainting')
args = parser.parse_args()

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

def save_black_white_mask(i, masks):
    mask = (masks[0] * 255)

    mask_save_path = './' + args.data + '_mask/'
    if not os.path.exists(mask_save_path):
        os.makedirs(mask_save_path)

    cv2.imwrite(mask_save_path + '{:05d}.png'.format(i), mask)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# setup Siammask
args.config = 'Siammask/get_mask/experiments/siammask/config_davis.json'
cfg = load_config(args)
siammask = Custom(anchors=cfg['anchors'])
if args.resume:
    assert isfile(args.resume), '{} is not a valid file'.format(args.resume)
    siammask = load_pretrain(siammask, args.resume)
siammask.eval().to(device)

# setup HQ-SAM
sam_checkpoint = "Model_CP/sam_hq_vit_tiny.pth"
model_type = "vit_tiny"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

# read images
images = [cv2.imread(image) for image in glob.glob(args.data + "/*.jpg")]

# selec target
cv2.namedWindow("mask", cv2.WINDOW_NORMAL)
x, y, w, h = cv2.selectROI("mask", images[0], showCrosshair=False, fromCenter=False)
cv2.destroyAllWindows()

# initialize Siammask tracking
target_pos = np.array([x + w / 2, y + h / 2])
target_sz = np.array([w, h])
state = siamese_init(images[0], target_pos, target_sz, siammask, cfg['hp'])  # init tracker

for i, image in enumerate(images):
    # Siammask tracking
    state = siamese_track(state, image, mask_enable=True, refine_enable=True)
    location = state['ploygon'].flatten()
    temp_location = np.reshape(np.intp(location), (4, -1))
    x_min0, y_min0 = np.min(temp_location, axis=0)
    x_max0, y_max0 = np.max(temp_location, axis=0)

    siammask_mask = state['mask'] > state['p'].seg_thr
    x_min1, y_min1, x_max1, y_max1 = create_box_from_mask(siammask_mask)

    x_min = (x_min0 + x_min1) / 2
    y_min = (y_min0 + y_min1) / 2
    x_max = (x_max0 + x_max1) / 2
    y_max = (y_max0 + y_max1) / 2

    # bounding box for predicting mask
    input_box = np.array([[x_min, y_min, x_max, y_max]])

    # HQ-SAM predicting mask inside of bounding box
    predictor.set_image(image)
    masks, _, _ = predictor.predict(
    point_coords=None,
    point_labels=None,
    box=input_box[None, :],
    multimask_output=False,
    )

    # visualize masked image
    show_masked_image(masks, input_box, image)
    # save mask for later use
    save_black_white_mask(i, masks)
plt.close()

# Run video inpaint
inpaint(args)
