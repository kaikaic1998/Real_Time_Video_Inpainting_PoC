import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob

#-----------------Siammask------------------
import argparse
from Siammask.get_mask.test import *
from Siammask.get_mask.models.custom import Custom
#-----------------Siammask------------------

from segment_anything_hq import sam_model_registry, SamPredictor

#-----------------Siammask------------------
parser = argparse.ArgumentParser(description='Demo')
parser.add_argument('--resume', default='Siammask/cp/SiamMask_DAVIS.pth', type=str,
                    metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--mask-dilation', default=32, type=int, help='mask dilation when inpainting')
args = parser.parse_args()
#-----------------Siammask------------------

def return_white_mask(mask):
    color = np.array([255, 255, 255])
    h, w = mask.shape[-2:]
    mask = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    return mask

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
    
    box = np.array([[min_x-10, min_y-10, min_x + box_width + 20, min_y + box_height + 20]])
    return box

def show_res_image(i, masks, scores, input_point, input_label, input_box, image):
    # image
    if input_box is not None:
        box = input_box[0]
        show_box(box, image)
    cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
    cv2.imshow('mask', image)
    # cv2.imwrite('C:/Users/Kainian/Desktop/WorkSpace/IM_Ghost_Project/images/annotation/{:05d}.png'.format(i), image)
    cv2.waitKey(1)

def show_res(i, masks, scores, input_point, input_label, input_box, image):
    # mask
    mask = return_white_mask(masks[0])
    if input_box is not None:
        box = input_box[0]
        show_box(box, mask)
    # visualize
    plt.gca().imshow(mask)
    plt.axis('on')
    plt.pause(0.001)
    cv2.imwrite('C:/Users/Kainian/Desktop/WorkSpace/IM_Ghost_Project/images/annotation/{:05d}.png'.format(i), mask)

device = "cuda"

#-----------------Siammask------------------
# Setup device
args.config = 'Siammask/get_mask/experiments/siammask/config_davis.json'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True

# Setup Model
cfg = load_config(args)
siammask = Custom(anchors=cfg['anchors'])
if args.resume:
    assert isfile(args.resume), '{} is not a valid file'.format(args.resume)
    siammask = load_pretrain(siammask, args.resume)

siammask.eval().to(device)
#-----------------Siammask------------------

sam_checkpoint = "C:/Users/Kainian/Desktop/WorkSpace/sam-hq/segment_anything/sam_hq_vit_h.pth"
model_type = "vit_h"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

images = [cv2.imread(image) for image in glob.glob("images/bmx-trees/*.jpg")]

cv2.namedWindow("mask", cv2.WINDOW_NORMAL)
x, y, w, h = cv2.selectROI("mask", images[0], showCrosshair=False, fromCenter=False)
cv2.destroyAllWindows()

#-----------------Siammask------------------
target_pos = np.array([x + w / 2, y + h / 2])
target_sz = np.array([w, h])
state = siamese_init(images[0], target_pos, target_sz, siammask, cfg['hp'])  # init tracker
#-----------------Siammask------------------

input_box = np.array([[x, y, x + w, y + h]])

for i, image in enumerate(images):
    predictor.set_image(image)
    input_point, input_label = None, None
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        box = input_box,
        multimask_output=False,
        hq_token_only= False,
    )

    show_res_image(i, masks,scores,input_point, input_label, input_box, image)
    # show_res(i, masks,scores,input_point, input_label, input_box, image)

    state = siamese_track(state, image, mask_enable=True, refine_enable=True)  # track
    location = state['ploygon'].flatten()
    temp_location = np.reshape(np.intp(location), (4, -1))
    x_min, y_min = np.min(temp_location, axis=0)
    x_max, y_max = np.max(temp_location, axis=0)

    input_box = np.array([[x_min, y_min, x_max, y_max]])

    # cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color=(0,255,0), thickness=2)
    # cv2.imshow('Get_mask', image)
    # cv2.waitKey(1)

    # input_box = create_box_from_mask(masks[0])

    # if i == 1:
    #     break

