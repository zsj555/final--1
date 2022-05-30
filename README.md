# final--1
## 一、将网上下载好的行车视频命名为video.mp4放入mmsegmentation-master文件夹内。
## 二、将在官网下载的模型fcn_s101-d8_512x1024_80k_cityscapes_20200807_140631-f8d155b3.pth放入mmsegmentation-master文件夹内。
## 三、开始测试
### config_file指向模型对应的网络结构
```
config_file = 'configs/resnest/fcn_s101-d8_512x1024_80k_cityscapes.py'
```
### checkpoint_file指向模型地址
```
checkpoint_file = 'fcn_s101-d8_512x1024_80k_cityscapes_20200807_140631-f8d155b3.pth'
```

### 逐帧测试原视频并生成图片
```
video = mmcv.VideoReader('video.mp4')  # 下载好的行车视频
i=0
for frame in video:
    i=i+1
    result = inference_segmentor(model, frame)
    model.show_result(frame, result, out_file='img/result/{}.jpg'.format(i))  # 将测试好的视频转化为一帧一帧的图片并存到img/result/文件夹中
 ```
 
 ### 将得到的图片合成为视频
