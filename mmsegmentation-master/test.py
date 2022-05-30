from mmseg.apis import inference_segmentor, init_segmentor
import mmcv
import os
import cv2

config_file = 'configs/resnest/fcn_s101-d8_512x1024_80k_cityscapes.py'
checkpoint_file = 'fcn_s101-d8_512x1024_80k_cityscapes_20200807_140631-f8d155b3.pth'

# 从一个 config 配置文件和 checkpoint 文件里创建分割模型
model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

# 测试一张样例图片并得到结果
# img = 'test.jpg'  # 或者 img = mmcv.imread(img), 这将只加载图像一次．
# result = inference_segmentor(model, img)
# 在新的窗口里可视化结果
# model.show_result(img, result, show=True)
# 或者保存图片文件的可视化结果
# 您可以改变 segmentation map 的不透明度(opacity)，在(0, 1]之间。
# model.show_result(img, result, out_file='result.jpg', opacity=0.5)

# 测试一个视频并得到分割结果

video = mmcv.VideoReader('video.mp4')
i=0
for frame in video:
    i=i+1
    result = inference_segmentor(model, frame)
    model.show_result(frame, result, out_file='img/result/{}.jpg'.format(i))





path = "img/result"#文件路径
filelist = os.listdir(path) #获取该目录下的所有文件名

fps = 30
size = (2304, 1440)  # 图片的分辨率
file_path = 'img/v' + ".mp4"  # 导出路径
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 不同视频编码对应不同视频格式（例：'I','4','2','0' 对应avi格式）

video = cv2.VideoWriter(file_path, fourcc, fps, size)

for item in filelist:
    if item.endswith('.jpg'):  # 判断图片后缀是否是.png
        item = path + '/' + item
        img = cv2.imread(item)  # 使用opencv读取图像，直接返回numpy.ndarray 对象，通道顺序为BGR ，注意是BGR，通道值默认范围0-255。
        video.write(img)  # 把图片写进视频

video.release()  # 释放
cv2.destroyAllWindows()



