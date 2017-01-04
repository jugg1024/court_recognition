### 下载编译 ###
0. 下载
git clone --recuersive git@github.com:jugg1024/court_recognition.git

1. 编译mqdf, 识别模块，得到OCRTool
cd mqdf && make

2. 编译caffe-fast-rcnn，注意这里caffe有变动，需重新编译
cd Text-Detection-with-FRCN/py-faster-rcnn/caffe-fast-rcnn （配置caffe环境，写Makefile.config)
make -j16 && make pycaffe

3. 编译fast-rcnn lib
cd Text-Detection-with-FRCN/py-faster-rcnn/lib
make

### 准备模型数据 ###
检测模型，检测proto，识别模型，贴图字体，都在data目录下：
/home/ligen/court_recognition/data/test.prototxt \  #检测prototxt
/home/ligen/court_recognition/data/vgg16_faster_rcnn_on_court_img_iter_100000.caffemodel \ #检测weight
/home/ligen/court_recognition/mqdf/OCRTool \  #识别工具
/home/ligen/court_recognition/data/template_bimoment_chinese_4_1230train12_3_.dat \  #识别模型

### demo ###
1. 运行脚本1，./script/court_img_recognition.sh，可以可视化识别结果的脚本:
./python/court_video_text_detect.py \
    --data_type images \   #处理数据类型，可以使imags和videos
    -if $1 \              #MP4文件或者jpg文件的目录
    -of ./output/$2 \     #输出文件目录
    --gpu 0 \             #gpu_id
    --det_prototxt /home/ligen/court_recognition/data/test.prototxt \  #检测prototxt
    --det_model  /home/ligen/court_recognition/data/vgg16_faster_rcnn_on_court_img_iter_100000.caffemodel \ #检测weight
    --ocr_tool  /home/ligen/court_recognition/mqdf/OCRTool \  #识别工具
    --ocr_model /home/ligen/court_recognition/data/template_bimoment_chinese_4_1230train12_3_.dat \  #识别模型
    --limit 5 \  #单个视频输出有文字帧的上限
    --interval 20 \  #采样帧的间隔
    --recognize 1 \  #是否识别
    --visualize 1    #是否可视化

2. 封装类，及api，参考./python/court_rec.py

   det_model = '/home/ligen/court_recognition/data/vgg16_faster_rcnn_on_court_img_iter_100000.caffemodel'
   det_model_proto = '/home/ligen/court_recognition/data/test.prototxt'
   rec_model = '/home/ligen/court_recognition/data/template_bimoment_chinese_4_1230train12_3_.dat'
   rec_tool = '/home/ligen/court_recognition/mqdf/OCRTool'
   gpu_id = 0 # -1 stand for cpu
   # 初始化类，输入以上五个参数，分别是模型数据以及是否用gpu
   cr = CourtRecognizor(det_model, det_model_proto, rec_model, rec_tool, gpu_id)
   # 输入为im和框的二维数组im, b_rects, 输出为精确框以及识别结果bboxs，reg_results
   im = cv2.imread('/home/ligen/court_recognition/demo/083_person_15_51s_grid24_bin17.jpg')
   h, w, _ = im.shape
   b_rects = [[0, 0, w, h]]  # left top width height
   bboxs, reg_results = cr.process(im, b_rects)
   print bboxs
   print reg_results

