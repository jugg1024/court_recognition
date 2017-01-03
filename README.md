0. 下载
git clone --recuersive git@github.com:jugg1024/court_recognition.git
1. 编译mqdf
cd mqdf && make
2. 编译caffe-fast-rcnn，注意这里caffe有变动，需重新编译
cd Text-Detection-with-FRCN/py-faster-rcnn/caffe-fast-rcnn （配置caffe环境，写Makefile.config)
make -j16 && make pycaffe
3. 编译fast-rcnn lib
cd Text-Detection-with-FRCN/py-faster-rcnn/lib
make
4. 准备模型数据
检测模型，检测proto，识别模型，贴图字体
5. 运行脚本
./script/court_img_recognition.sh
   --data_type images \
    -if $1 \
    -of ./output/$2 \
    --gpu 0 \
    --det_prototxt /home/ligen/court_recognition/data/test.prototxt \
    --det_model  /home/ligen/court_recognition/data/vgg16_faster_rcnn_on_court_img_iter_100000.caffemodel \
    --ocr_tool  /home/ligen/court_recognition/mqdf/OCRTool \
    --ocr_model /home/ligen/court_recognition/data/template_bimoment_chinese_4_1230train12_3_.dat \
    --limit 5 \
    --interval 20 \
    --recognize 1 \
    --visualize 1
参数最好写绝对路径
