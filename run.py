import cv2
import glob
import argparse

# import utility
from utils import *

# import Siammask modules
from Siammask.test import *
from Siammask.models.custom import Custom

# import Deep Video Inpainting modules
from Deep_Video_Inpainting.inpaint import inpaint

# import HQ_SAM modules
from segment_anything_hq import sam_model_registry, SamPredictor

parser = argparse.ArgumentParser(description='Demo')

parser.add_argument('--resume', default='Model_CP/SiamMask_DAVIS.pth', type=str, metavar='PATH')
parser.add_argument('--data', default='data/car-turn')
parser.add_argument('--mask-dilation', default=32, type=int, help='mask dilation when inpainting')
args = parser.parse_args()

# when input is video file
if args.data[-4:] == '.mp4':
    create_images_from_video(args.data)
    args.data = args.data[:-4]

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# setup Siammask
args.config = 'Siammask/experiments/siammask/config_davis.json'
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
cv2.namedWindow("Select Target", cv2.WINDOW_NORMAL)
x, y, w, h = cv2.selectROI("Select Target", images[0], showCrosshair=False, fromCenter=False)
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
    save_black_white_mask(i, masks, args.data)
plt.close()

# Run video inpaint
inpaint(args)
