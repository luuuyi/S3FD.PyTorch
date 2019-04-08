from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.config_s3fd import cfg
from layers.functions.prior_box_s3fd import PriorBox
from utils.nms_wrapper import nms
import cv2
from models.s3fd import S3FD
from utils.box_utils import decode
from utils.timer import Timer
import scipy.io as sio
import pdb

parser = argparse.ArgumentParser(description='S3FD')

parser.add_argument('-m', '--trained_model', default='../weights/S3FD_test/Final_S3FD.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--input', default='./', type=str, help='Input Image Dir')
parser.add_argument('--save', default='tmp/', type=str, help='Dir to save results')
parser.add_argument('--cuda', default=True, type=bool, help='Use cuda to train model')
parser.add_argument('--cpu', default=False, type=bool, help='Use cpu nms')
parser.add_argument('--confidence_threshold', default=0.05, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.3, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
args = parser.parse_args()


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path):
    print('Loading pretrained model from {}'.format(pretrained_path))
    device = torch.cuda.current_device()
    pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

def detect_face(net, img, resize):
    if resize != 1:
        img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
    im_height, im_width, _ = img.shape
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    #img = img[[2, 1, 0], :, :]
    img = torch.from_numpy(img).unsqueeze(0)
    if args.cuda:
        img = img.cuda()
        scale = scale.cuda()

    out = net(img)  # forward pass
    priorbox = PriorBox(cfg, out[2], (im_height, im_width), phase='test')
    priors = priorbox.forward()
    if args.cuda:
        priors = priors.cuda()
    loc, conf, _ = out
    print(loc.size(), conf.size())
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale / resize
    boxes = boxes.cpu().numpy()
    scores = conf.data.cpu().numpy()[:, 1]

    # ignore low scores
    inds = np.where(scores > args.confidence_threshold)[0]
    boxes = boxes[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:args.top_k]
    boxes = boxes[order]
    scores = scores[order]
    #print(boxes)

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = nms(dets, args.nms_threshold, force_cpu=args.cpu)
    dets = dets[keep, :]
    #print(dets)

    # keep top-K faster NMS
    dets = dets[:args.keep_top_k, :]
    return dets

def flip_test(net, image, shrink):
    image_f = cv2.flip(image, 1)
    det_f = detect_face(net, image_f, shrink)

    det_t = np.zeros(det_f.shape)
    det_t[:, 0] = image.shape[1] - det_f[:, 2]
    det_t[:, 1] = det_f[:, 1]
    det_t[:, 2] = image.shape[1] - det_f[:, 0]
    det_t[:, 3] = det_f[:, 3]
    det_t[:, 4] = det_f[:, 4]
    return det_t

def multi_scale_test(net, image, max_im_shrink):
    # shrink detecting and shrink only detect big face
    st = 0.5 if max_im_shrink >= 0.75 else 0.5 * max_im_shrink
    det_s = detect_face(net, image, st)
    index = np.where(np.maximum(det_s[:, 2] - det_s[:, 0] + 1, det_s[:, 3] - det_s[:, 1] + 1) > 30)[0]
    det_s = det_s[index, :]

    # enlarge one times
    bt = min(2, max_im_shrink) if max_im_shrink > 1 else (st + max_im_shrink) / 2
    det_b = detect_face(net, image, bt)

    # enlarge small iamge x times for small face
    if max_im_shrink > 2:
        bt *= 2
        while bt < max_im_shrink:
            det_b = np.row_stack((det_b, detect_face(net, image, bt)))
            bt *= 2
        det_b = np.row_stack((det_b, detect_face(net, image, max_im_shrink)))

    # enlarge only detect small face
    if bt > 1:
        index = np.where(np.minimum(det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1) < 100)[0]
        det_b = det_b[index, :]
    else:
        index = np.where(np.maximum(det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1) > 30)[0]
        det_b = det_b[index, :]

    return det_s, det_b

def bbox_vote(det):
    order = det[:, 4].ravel().argsort()[::-1]
    det = det[order, :]
    while det.shape[0] > 0:
        # IOU
        area = (det[:, 2] - det[:, 0] + 1) * (det[:, 3] - det[:, 1] + 1)
        xx1 = np.maximum(det[0, 0], det[:, 0])
        yy1 = np.maximum(det[0, 1], det[:, 1])
        xx2 = np.minimum(det[0, 2], det[:, 2])
        yy2 = np.minimum(det[0, 3], det[:, 3])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o = inter / (area[0] + area[:] - inter)

        # get needed merge det and delete these det
        merge_index = np.where(o >= 0.3)[0]
        det_accu = det[merge_index, :]
        det = np.delete(det, merge_index, 0)

        if merge_index.shape[0] <= 1:
            continue
        det_accu[:, 0:4] = det_accu[:, 0:4] * np.tile(det_accu[:, -1:], (1, 4))
        max_score = np.max(det_accu[:, 4])
        det_accu_sum = np.zeros((1, 5))
        det_accu_sum[:, 0:4] = np.sum(det_accu[:, 0:4], axis=0) / np.sum(det_accu[:, -1:])
        det_accu_sum[:, 4] = max_score
        try:
            dets = np.row_stack((dets, det_accu_sum))
        except:
            dets = det_accu_sum

    dets = dets[0:750, :]
    return dets

def draw_rect(image, det):
    for i in range(det.shape[0]):
        xmin = int(det[i][0])
        ymin = int(det[i][1])
        xmax = int(det[i][2])
        ymax = int(det[i][3])
        score = det[i][4]
        if score > 0.1:
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 4)
        #cv2.putText(image, str(score), (int(xmin), int(ymin) - 5), 
        #            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)

if __name__ == '__main__':
    # net and model
    net = S3FD(phase='test', size=None, num_classes=2)    # initialize detector
    net = load_model(net, args.trained_model)
    net.eval()
    print('Finished loading model!')
    print(net)
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    else:
        net = net.cpu()

    assert args.input != None, 'Input Image Dir Should not be None'
    if not os.path.isdir(args.save):
        os.makedirs(args.save)

    img_paths = [os.path.join(args.input, x) for x in os.listdir(args.input) if x.endswith('jpg')]
    for index, _img_path in enumerate(img_paths):
        Image_Path = _img_path
        assert os.path.isfile(Image_Path), 'File not exit: %s' % (Image_Path)
        image = np.float32(cv2.imread(Image_Path, cv2.IMREAD_COLOR))
        image_t = image.copy()
        max_im_shrink = (0x7fffffff / 577.0 / (image.shape[0] * image.shape[1])) ** 0.5 # the max size of input image for PyTorch
        #max_im_shrink = np.sqrt(1700 * 1200 / (image.shape[0] * image.shape[1]))
        resize = max_im_shrink if max_im_shrink < 1 else 1
        #dets = detect_face(net, image, max_im_shrink)
        dets = detect_face(net, image, resize)

        '''det0 = detect_face(net, image, resize)  # origin test
        det1 = flip_test(net, image, resize)    # flip test
        [det2, det3] = multi_scale_test(net, image, max_im_shrink)  #multi-scale test

        # merge all test results via bounding box voting
        det = np.row_stack((det0, det1, det2, det3))
        dets = bbox_vote(det)'''

        draw_rect(image_t, dets)
        save_path = os.path.join(args.save, _img_path.replace('.jpg', '_result.jpg'))
        cv2.imwrite(save_path, image_t)
        print('Process No.%d Done!' % (index))
