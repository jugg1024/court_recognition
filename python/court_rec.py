#!/usr/bin/python
# -*- coding: UTF-8 -*-
import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2, glob

class CourtRecognizor:
   def __init__(self, det_model, det_model_proto, rec_model, rec_tool, gpu_id):
      self.det_net = self.detect_init(det_model_proto, det_model, gpu_id)
      self.ocr_tool = rec_tool
      self.ocr_model = rec_model
   
   def detect_init(self, prototxt, caffemodel, gpu_id):
      cfg.TEST.HAS_RPN = True  # Use RPN for proposals
      if not os.path.isfile(caffemodel):
         raise IOError(('{:s} not found.\nDid you run ./data/script/'
               'fetch_faster_rcnn_models.sh?').format(caffemodel))
      if gpu_id == -1:
         caffe.set_mode_cpu()
      else:
         caffe.set_mode_gpu()
         caffe.set_device(gpu_id)
         cfg.GPU_ID = gpu_id
      net = caffe.Net(prototxt, caffemodel, caffe.TEST)
      print '\n\nLoaded network {:s}'.format(caffemodel)
      # Warmup on a dummy image
      im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
      for i in xrange(2):
         _, _= im_detect(net, im)
      return net

   def im_det(self, im):
     # Detect all text in a Image
     CONF_THRESH = 0.5
     scores, boxes = im_detect(self.det_net, im)
     cls_scores = scores[:, 1]
     cls_boxes = boxes[:, 4:8]
     dets = np.hstack((cls_boxes,
             cls_scores[:, np.newaxis])).astype(np.float32)
     inds = np.where(dets[:,-1] >= CONF_THRESH)[0]
     dets = dets[inds, :]
     return dets

   def im_rec(self, imname):
      exestr = self.ocr_tool + ' ' + self.ocr_model + ' ' + imname
      res = os.popen(exestr).read()
      print 'res is', res
      return res

   def process(self, img, b_rects):
      h, w, _ = img.shape
      bboxs = np.array([], dtype=np.int64).reshape(0, 5)
      for rect in b_rects:
         im = img[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2], :]
         dets = self.im_det(im)
         dets[:, 0] = dets[:, 0] + rect[0]
         dets[:, 2] = dets[:, 2] + rect[0]
         dets[:, 1] = dets[:, 1] + rect[1]
         dets[:, 3] = dets[:, 3] + rect[1]
         bboxs = np.vstack((bboxs, dets)).astype(np.float32)
      NMS_THRESH = 0.2
      keep = nms(bboxs, NMS_THRESH)
      bboxs = bboxs[keep, :]
      padding = 5
      reg_results = []
      for idx, det in enumerate(bboxs):
         top = np.max([int(det[1]) - padding, 0])
         bottom = np.min([int(det[3]) + padding, h])
         left = np.max([int(det[0]) - padding, 0])
         right = np.min([int(det[2]) + padding, w])
         patch_name = 'temp' + str(idx) + '.png'
         im2 = img[top:bottom, left:right, :]
         cv2.imwrite(patch_name, im2)
         res = self.im_rec(patch_name)
         reg_results.append(res)
      return bboxs, reg_results

if __name__ == '__main__':
   det_model = '/home/ligen/court_recognition/data/vgg16_faster_rcnn_on_court_img_iter_100000.caffemodel'
   det_model_proto = '/home/ligen/court_recognition/data/test.prototxt'
   rec_model = '/home/ligen/court_recognition/data/template_bimoment_chinese_4_1230train12_3_.dat'
   rec_tool = '/home/ligen/court_recognition/mqdf/OCRTool'
   gpu_id = 0 # -1 stand for cpu
   cr = CourtRecognizor(det_model, det_model_proto, rec_model, rec_tool, gpu_id)
   im = cv2.imread('/home/ligen/court_recognition/demo/083_person_15_51s_grid24_bin17.jpg')
   h, w, _ = im.shape
   b_rects = [[0, 0, w, h]]  # left top width height
   bboxs, reg_results = cr.process(im, b_rects)
   print bboxs
   print reg_results