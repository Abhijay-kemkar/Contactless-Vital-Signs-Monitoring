from config import *
from models.retinaface.data.config import cfg_mnet, cfg_re50
from models.retinaface.layers.functions.prior_box import PriorBox
from models.retinaface.utils.nms.py_cpu_nms import py_cpu_nms
from models.retinaface.models.retinaface import RetinaFace
from models.retinaface.utils.box_utils import decode, decode_landm
from models.retinaface.utils.load_model import *
import torch
import torch.backends.cudnn as cudnn
import cv2
import numpy as np
from torch2trt import torch2trt



class Detector:

    def __init__(self, face_detection_resolution=FACE_DETECTION_RESOLUTION, cpu=CPU, network_backbone=NETWORK_TYPE, trained_model_res=TRAINED_MODEL_RES,
                 trained_model_mobile=TRAINED_MODEL_MOBILE,
                 confidence_threshold=CONFIDENCE_THRESHOLD, keep_top_k=KEEP_TOP_K):
        self.network_backbone = network_backbone
        self.trained_model_res = trained_model_res
        self.trained_model_mobile = trained_model_mobile
        self.cpu = cpu
        #you can also set face_detection_resolution to None, then it will take the normal resolution
        self.face_detection_resolution = face_detection_resolution

        self.confidence_threshold = confidence_threshold
        self.keep_top_k = keep_top_k

        # get cofniguration for network
        if network_backbone == "mobile":
            self.cfg = cfg_mnet
            self.trained_model = trained_model_mobile
        elif network_backbone == "resnet":
            self.cfg = cfg_re50
            self.trained_model = trained_model_res
        else:
            print('Invalid model: Choose from "mobile0.25" or "resnet50" ')

        # get model
        self.getModel()

    def getModel(self):
        self.net = RetinaFace(cfg=self.cfg, phase='test')
        self.net = load_model(self.net, self.trained_model, self.cpu)
        self.net.eval()

    def loadToDevice(self, device, toTensorRT = True,):
        cudnn.benchmark = True
        self.device = device
        self.net = self.net.to(self.device)
        if toTensorRT:
            dim_1, dim_2 = self.face_detection_resolution[0], self.face_detection_resolution[1]
            x = torch.ones((1, 3, dim_1, dim_2)).cuda()
            self.net = torch2trt(self.net, [x])

    def forward(self, img_raw):
        # preprocess img
        img = self._preprocess(img_raw)
        # forward pass
        location, confidence, _ = self.net(img)
        # postprocess
        bbox = self._postprocess( location, confidence)
        return bbox

    def _preprocess(self, img_raw):
        #get original img resolution
        self.img_raw_resolution = img_raw.shape

        # pre image conversion
        img = np.float32(img_raw)
        if self.face_detection_resolution:
            img = cv2.resize(img, self.face_detection_resolution, interpolation=cv2.INTER_LINEAR)

        #if resize != 1:
        im_height, im_width, _ = img.shape
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)

        # get shape
        self.shape = (im_height, im_width)

        return img

    def _postprocess(self, location, confidence):
        # post box processing after forward pass
        priorbox = PriorBox(self.cfg, image_size=self.shape)
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        boxes = decode(location.data.squeeze(0), prior_data, self.cfg['variance'])

        #scale tensor to upscale boxes to orignal resolution
        scale = torch.tensor([self.img_raw_resolution[1], self.img_raw_resolution[0], self.img_raw_resolution[1], self.img_raw_resolution[0]])
        scale = scale.to(self.device)
        boxes = boxes * scale
        boxes = boxes.cpu().numpy()
        scores = confidence.squeeze(0).data.cpu().numpy()[:, 1]

        #only keep boxes with a given confidence score
        inds = np.where(scores > self.confidence_threshold)[0]
        boxes = boxes[inds]
        scores = scores[inds]

        # keep top-K before NMS
        # order = scores.argsort()[::-1][:args.top_k]
        order = scores.argsort()[::-1]
        boxes = boxes[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.confidence_threshold)
        dets = dets[keep, :]

        # keep top-K faster NMS
        dets = dets[:self.keep_top_k, :]

        return dets
