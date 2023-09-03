# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import glob
import argparse
from Siammask.get_mask.test import *
from Siammask.get_mask.models.custom import Custom

parser = argparse.ArgumentParser(description='Demo')

parser.add_argument('--resume', default='Siammask/cp/SiamMask_DAVIS.pth', type=str,
                    metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--mask-dilation', default=32, type=int, help='mask dilation when inpainting')
args = parser.parse_args()

def mask(args):
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

    ims = [cv2.imread(image) for image in glob.glob("images/bmx-trees/*.jpg")]

    # Select ROI
    cv2.namedWindow("Get_mask", cv2.WND_PROP_FULLSCREEN)
    # cv2.setWindowProperty("SiamMask", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    x, y, w, h = cv2.selectROI('Get_mask', ims[0], False, False)

    target_pos = np.array([x + w / 2, y + h / 2])
    target_sz = np.array([w, h])
    state = siamese_init(ims[0], target_pos, target_sz, siammask, cfg['hp'])  # init tracker

    toc = 0
    counter = 0

    for i, im in enumerate(ims):
        tic = cv2.getTickCount()

        state = siamese_track(state, im, mask_enable=True, refine_enable=True)  # track
        location = state['ploygon'].flatten()

        mask = state['mask'] > state['p'].seg_thr
        mask = (mask * 255.).astype(np.uint8)
        cv2.imwrite('Siammask/results/data/bmx-trees_mask/{:05d}.png'.format(counter), mask)

        im[:, :, 2] = (mask > 0) * 255 + (mask == 0) * im[:, :, 2]

        # cv2.polylines(im, [np.intp(location).reshape((-1, 1, 2))], True, (0, 255, 0), 3)

        temp_location = np.reshape(np.intp(location), (4, -1))
        max_point = np.max(temp_location, axis=0)
        min_point = np.min(temp_location, axis=0)

        cv2.rectangle(im, min_point, max_point, color=(0,255,0), thickness=2)
        cv2.imwrite('Siammask/results/data/bmx-trees_frame/{:05d}.jpg'.format(counter), im)
        cv2.imshow('Get_mask', im)
        cv2.waitKey(1)

        toc += cv2.getTickCount() - tic

        counter += 1

        if i == 0:
            break

    toc /= cv2.getTickFrequency()
    fps = i / toc
    print('SiamMask Time: {:02.1f}s Speed: {:3.1f}fps (with visulization!)'.format(toc, fps))
    cv2.destroyAllWindows()

mask(args)

