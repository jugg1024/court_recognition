#!/usr/bin/env python

# --------------------------------------------------------
# Court Img Detection and Recognition
# Copyright (c) 2017
# Written by Gen Li
# --------------------------------------------------------

"""
Demo script showing text detections in sample images.
See README.md for installation instructions before running.
"""
import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2, glob
import argparse
import Image
import ImageFont, ImageDraw
from bs4 import BeautifulSoup

from os import listdir
from os.path import isfile, join

grid = [[2, 3], [3, 3], [3, 4]]
font_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data/simhei.ttf')
font = ImageFont.truetype(font_path, 20)

def parse_args():
  """Parse input arguments."""
  parser = argparse.ArgumentParser(description='Faster R-CNN demo')
  parser.add_argument('--data_type',
                      choices = ['video', 'videos', 'image', 'images'],
                      help = 'what kind of data to process')
  parser.add_argument('-if', '--input_folder',
                      dest='input_folder', 
                      help='input folder name',
                      default='./')
  parser.add_argument('-of', '--output_folder',
                      dest='output_folder',
                      help='output folder name',
                      default=os.path.join(cfg.ROOT_DIR, 'result'))
  parser.add_argument('-vu', '--video_uri', 
                      dest='video_uri', 
                      help='video uri',
                      default='./001_heise.mp4')
  parser.add_argument('-iu', '--image_uri', 
                      dest='image_uri', 
                      help='image uri',
                      default='./test.jpg')
  parser.add_argument('--gpu', 
                      dest='gpu_id', 
                      help='GPU device id to use [0]',
                      default=1, 
                      type=int)
  parser.add_argument('--cpu', 
                      dest='cpu_mode',
                      help='Use CPU mode (overrides --gpu)',
                      action='store_true')
  parser.add_argument('--det_prototxt', 
                      dest='prototxt', 
                      help='Network prototxt')
  parser.add_argument('--det_model', 
                      dest='trained_model',
                      help='weights')
  parser.add_argument('--ocr_tool', 
                      dest='ocr_tool',
                      help='ocr_tool')  
  parser.add_argument('--ocr_model', 
                      dest='ocr_model',
                      help='ocr_model')
  parser.add_argument('-l', '--limit', 
                      dest='limit',
                      help='maximum number of frame per video',
                      default=10,
                      type=int)
  parser.add_argument('--interval', 
                      dest='interval',
                      help='detect every [interval] seconds',
                      default=10,
                      type=int)
  parser.add_argument('--recognize', 
                      dest='recognize',
                      help='recognize or not',
                      default=1,
                      type=int)
  parser.add_argument('--visualize', 
                      dest='visualize',
                      help='visualize or not',
                      default=1,
                      type=int)
  parser.add_argument('--eval', 
                      dest='eval',
                      help='eval or not',
                      default=0,
                      type=int)
  args = parser.parse_args()
  return args

def caffe_init(args):
  cfg.TEST.HAS_RPN = True  # Use RPN for proposals
  prototxt = args.prototxt
  if args.trained_model:
    caffemodel = args.trained_model

  if not os.path.isfile(caffemodel):
    raise IOError(('{:s} not found.\nDid you run ./data/script/'
             'fetch_faster_rcnn_models.sh?').format(caffemodel))

  if args.cpu_mode:
    caffe.set_mode_cpu()
  else:
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    cfg.GPU_ID = args.gpu_id
  net = caffe.Net(prototxt, caffemodel, caffe.TEST)
  print '\n\nLoaded network {:s}'.format(caffemodel)
  # Warmup on a dummy image
  im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
  for i in xrange(2):
    _, _= im_detect(net, im)
  return net

def draw_result(im_path, dets, ress, sec):
  image = Image.open(im_path)
  draw = ImageDraw.Draw(image)
  if sec >= 0:
    draw.text((20, 20), str(sec) + ' seconds', (255,0,0) ,font=font)
  for idx, det in enumerate(dets):
    x = det[0]
    y = det[1]
    text = unicode(ress[idx], "utf-8")
    draw.text((x, y-15), text,(255,0,0) ,font=font)
  image.save(im_path)

def patch_recognit(imname, args):
  #exestr = 'tesseract --tessdata-dir /usr/share/ ' + imname + ' ' + imname + ' -l chi_sim'
  # binary = '~/Text-Detection-with-FRCN/mqdf/OCRTool'
  # model = '~/Text-Detection-with-FRCN/mqdf/template_bimoment_chinese_4_1230train12_3_.dat'
  exestr = args.ocr_tool + ' ' + args.ocr_model + ' ' + imname
  res = os.popen(exestr).read()
  print 'res is', res
  return res
  
def im_det(net, im):
  # Detect all text in a Image
  CONF_THRESH = 0.5
  scores, boxes = im_detect(net, im)
  cls_scores = scores[:, 1]
  cls_boxes = boxes[:, 4:8]
  dets = np.hstack((cls_boxes,
          cls_scores[:, np.newaxis])).astype(np.float32)
  inds = np.where(dets[:,-1] >= CONF_THRESH)[0]
  dets = dets[inds, :]
  return dets

def frame_det(net, img):
  # Cutting Frame into bins, and detect in each of them
  h, w, _ = img.shape
  res = np.array([], dtype=np.int64).reshape(0, 5)
  for g in grid:
    bin_h = h / g[0]
    bin_w = w / g[1]
    for idx in xrange(g[0] * g[1]):
      im = img[idx / g[1] * bin_h:(idx / g[1] + 1) * bin_h, idx % g[1] * bin_w:(idx % g[1] + 1) * bin_w, :]
      dets = im_det(net, im)
      dets[:, 0] = dets[:, 0] + idx % g[1] * bin_w
      dets[:, 2] = dets[:, 2] + idx % g[1] * bin_w
      dets[:, 1] = dets[:, 1] + idx / g[1] * bin_h
      dets[:, 3] = dets[:, 3] + idx / g[1] * bin_h
      res = np.vstack((res, dets)).astype(np.float32)
      np.concatenate((res, dets), axis=0)
  NMS_THRESH = 0.2
  keep = nms(res, NMS_THRESH)
  res = res[keep, :]
  return res

def check_frame_valid(frame, det_cnt, frame_cnt, fps, args):
  # -1: bad frame
  # 0: normal frame
  # 1: good frame
  if (type(frame) == type(None)):
    print 'Image Type is None'
    return -1
  elif (0xFF & cv2.waitKey(5) == 27) or frame.size == 0:
    print 'Image Size is Zero'
    return -1
  elif det_cnt >= args.limit:
    print 'Text is detected in more than '+ str(args.limit) + ' frame, Stop.'
    return -1
  elif frame_cnt % (args.interval * fps) != 0:
    return 0
  else:
    return 1

def parse_gt_from_xml(xml_file):
  gt = open(xml_file, 'r')
  handle = BeautifulSoup(gt.read())
  labels = handle.annotation.findAll('object')
  reg_lab = []
  det_lab = []
  for label in labels:
    reg_lab.append(label.contents[1].contents[0])
    det_lab.append([int(label.xmin.contents[0]), int(label.ymin.contents[0]), 
                int(label.xmax.contents[0]), int(label.ymax.contents[0])])
  return [np.array(det_lab), reg_lab]

def parse_gt_from_txt(txt_file):
  gt = open(txt_file, 'r')
  det_lab = []
  reg_lab = []
  for line in gt:
    strs = line.strip().split()
    det_lab.append([int(strs[0]), int(strs[1]), 
                int(strs[2]) + int(strs[0]) - 1, int(strs[3]) + int(strs[1]) - 1])
  return [np.array(det_lab), reg_lab]

def compute_iou(dets, labels):
  # true pos + false pos = len(dets)
  # true pos + false neg = len(labels)
  #    tp fp fn
  m = [0, len(dets), len(labels)]
  if len(dets) == 0 or len(labels)==0:
    return m
  area_dets = (dets[:, 3] - dets[:, 1]) * (dets[:, 2] - dets[:, 0])
  area_labels = (dets[:, 3] - dets[:, 1]) * (dets[:, 2] - dets[:, 0])
  print dets
  print labels
  for label in labels:
    insect = dets.copy()
    insect[:, 0] = np.maximum(insect[:, 0], label[0])
    insect[:, 1] = np.maximum(insect[:, 1], label[1])
    insect[:, 2] = np.minimum(insect[:, 2], label[2])
    insect[:, 3] = np.minimum(insect[:, 3], label[3])
    area_insect = (insect[:, 3] - insect[:, 1]) * (insect[:, 2] - insect[:, 0]) * (insect[:, 2] > insect[:, 0]) * (insect[:, 3] > insect[:, 1])
    area_union = area_dets + area_labels - area_insect + 0.0000001
    iou = np.max(area_insect / area_union)
    print iou
    if iou > 0.5:
      m[0] += 1
      m[1] -= 1
      m[2] -= 1
  return m

def image_det(image_uri, net, args):
  global grid
  #grid = [[1, 1], [2, 3], [3, 3]]
  grid = [[1, 1]]
  '''
  create image path in output folder
  '''
  detection_res_path = os.path.join(args.output_folder, 'det_res')
  recognize_res_path = os.path.join(args.output_folder, 'reg_res')
  if not os.path.isdir(detection_res_path):
    os.makedirs(detection_res_path)
  if args.recognize > 0:
    if not os.path.isdir(recognize_res_path):
      os.makedirs(recognize_res_path)
  prefix = image_uri.split('/')[-1][:-4]
  det_image_path = os.path.join(detection_res_path, prefix)
  img = cv2.imread(image_file)
  dets = frame_det(net, img)
  if args.eval > 0:
    #gt_path = image_uri[:-4] + '.xml'
    #[det_labs, reg_labs] = parse_gt_from_xml(gt_path)
    gt_path = image_uri + '.txt'
    [det_labs, reg_labs] = parse_gt_from_txt(gt_path)
    m = compute_iou(dets, det_labs)

  reg_results = []
  if args.recognize > 0:
    h, w, _ = img.shape
    padding = 5
    reg_image_path = os.path.join(recognize_res_path, prefix)
    if not os.path.isdir(reg_image_path):
      os.makedirs(reg_image_path)
    for idx, det in enumerate(dets):
      top = np.max([int(det[1]) - padding, 0])
      bottom = np.min([int(det[3]) + padding, h])
      left = np.max([int(det[0]) - padding, 0])
      right = np.min([int(det[2]) + padding, w])
      patch_name = os.path.join(reg_image_path, 'patch' + str(idx) + '.png')
      im2 = img[top:bottom, left:right, :]
      cv2.imwrite(patch_name, im2)
      res = patch_recognit(patch_name, args)
      reg_results.append(res)
  if args.visualize > 0 :
    if args.eval == 0:
      frame_name = det_image_path + ".png"
    elif m[2] > 0 or m[1] > 0:
      frame_name = det_image_path + "bad.png"
    else:
      frame_name = det_image_path + "good.png"
    im = img.astype(np.uint8).copy()
    #for idx, det in enumerate(det_labs):
    #  cv2.rectangle(im, (int(det[0]), int(det[1])), (int(det[2]), int(det[3])), (0,255,0), 2)
    for idx, det in enumerate(dets):
      cv2.rectangle(im, (int(det[0]), int(det[1])), (int(det[2]), int(det[3])), (255,0,0), 2)
    cv2.imwrite(frame_name, im)
    if args.recognize > 0:
      draw_result(frame_name, dets, reg_results, -1)
  if args.eval > 0:
    return np.array(m)
  else: 
    return -1

def video_det(video_uri, net, args):
  '''
  create video path in output folder
  '''
  if args.visualize > 0:
    prefix = video_uri.split('/')[-1][:-4]
    video_path = os.path.join(args.output_folder, prefix)
    if not os.path.isdir(video_path):
      os.makedirs(video_path)

  '''
  video input
  '''
  capture = cv2.VideoCapture(video_uri)
  # if opencv 2.* then use cv2.cv.CV_CAP*
  # if opencv > 3.0.0 then use cv2.CAP*
  size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
          int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
  fps = int(capture.get(cv2.CAP_PROP_FPS) + 0.5)
  print 'video frame per second:', fps
  
  # frame count
  frame_cnt = 0
  # detect count
  det_cnt = 0
  
  '''
  detect text frame by frame with a interval
  '''
  # start timer
  vtimer = Timer()
  vtimer.tic()
  while True:
    # read frame
    ret, img = capture.read()
    # check frame validation
    valid = check_frame_valid(img, det_cnt, frame_cnt, fps, args)
    if valid < 0:
      break
    if valid == 0:
      pass
    if valid > 0:
      # good frame, start processing
      # start timer
      ftimer = Timer()
      ftimer.tic()
      # create frame folder
      seconds = frame_cnt / fps
      print 'processing', str(seconds), 'second frame'
      frame_path = os.path.join(video_path, prefix + '_' + str(seconds) + 's')
      if not os.path.isdir(frame_path):
        os.makedirs(frame_path)
      # det in frame
      dets = frame_det(net, img)
      if dets.shape[0] > 0:
        det_cnt += 1
        reg_results = []
        if args.recognize > 0:
          h, w, _ = img.shape
          padding = 0
          top = np.max([int(det[1]) - padding, 0])
          bottom = np.min([int(det[3]) + padding, h])
          left = np.max([int(det[0]) - padding, 0])
          right = np.max([int(det[2]) + padding, w])
          for idx, det in enumerate(dets):
            patch_name = os.path.join(frame_path, 'patch' + str(idx) + '.png')
            im2 = img[top:bottom, left:right, :]
            cv2.imwrite(patch_name, im2)
            res = patch_recognit(patch_name, args)
            reg_results.append(res)
        if args.visualize > 0:
          frame_name = os.path.join(video_path, prefix + '_' + str(seconds) + 's.png')
          im = img.astype(np.uint8).copy()
          for idx, det in enumerate(dets):
            cv2.rectangle(im, (int(det[0]), int(det[1])), (int(det[2]), int(det[3])), (0,255,0), 2)
          cv2.imwrite(frame_name, im)
          if args.recognize > 0:
            draw_result(frame_name, dets, reg_results, seconds)
      ftimer.toc()
      print ('Detection took {:.3f}s for '
        '{:d} text proposals').format(ftimer.total_time, dets.shape[0])
    frame_cnt += 1
  vtimer.toc()
  print ('Detection took {:.3f}s for '
         'video {}').format(vtimer.total_time, video_uri)
  capture.release()
  cv2.destroyAllWindows()
  
if __name__ == '__main__':
  # parse input
  args = parse_args()
  print 'input args:', args
  # caffe model init
  net = caffe_init(args)
  if args.data_type == 'videos':
    # detect text in video
    for video_file in glob.glob(args.input_folder + '/*.mp4'):
      video_det(video_file, net, args)
  elif args.data_type == 'video':
    # detect text in video
    video_det(args.video_uri, net, args)
  elif args.data_type == 'images':
    if args.eval > 0:
      matr = np.array([0, 0, 0])
      for image_file in glob.glob(args.input_folder + '/*.jpg'):
        m = image_det(image_file, net, args)
        matr += m
        print matr
        print 'precision', 1.0 * matr[0] / (matr[0] + matr[1] + 0.000001)
        print 'recall', 1.0 * matr[0] / (matr[0] + matr[2] + 0.000001)
    else:
      for image_file in glob.glob(args.input_folder + '/*.jpg'):
        m = image_det(image_file, net, args)
  elif args.data_type == 'image':
    print 'not implement'
  else:
    print 'not implement'
