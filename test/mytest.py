# -*- coding: UTF-8 -*-
"""
@version: 1.0
@PackageName: test - mytest.py
@author: junchen
@Description: 
@since 2024/01/23 12:37
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
#导入BosClient配置文件
import sts_sample
#导入BOS相关模块
from baidubce import exception
from baidubce.services import bos
from baidubce.services.bos import canned_acl
from baidubce.services.bos.bos_client import BosClient
import time
import requests
from requests.exceptions import RequestException
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, Response, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from flask import Flask, request, send_file, make_response
from flask_cors import CORS
import json
import urllib

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "https://yiyan.baidu.com"}})

bos_client = BosClient(sts_sample.config)


@app.route('/sam/predict', methods=['POST'])
async def samtest():
    # 加载SAM预训练模型
    import sys
    sys.path.append("..")  # 将上一级目录加入sys.path，这样才能执行下面一句
    from segment_anything import sam_model_registry, SamPredictor

    request_data = request.get_json()
    url = request_data.get('url')

    sam_checkpoint = 'checkpoint/sam_vit_b_01ec64.pth'
    model_type = "vit_b"
    device = "cuda"  # "cpu"  cuda

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    # 导入待分割图片
    image = None
    try:
        with request.get(url) as url_response:
            s = url_response.read()
        arr = np.asarray(bytearray(s), dtype=np.uint8)
        image = cv2.imdecode(arr, -1)  # 'Load it as it is'
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    except Exception as e:
        print("获取图片出错")

    predictor = SamPredictor(sam)
    # 编码图像
    predictor.set_image(image)

    # 单点 prompt  输入格式为(x, y)和并表示出点所带有的标签1(前景点)或0(背景点)。
    height, width = image.shape[:2]
    input_point = np.array([[height/2, width/2]])  # 标记点
    input_label = np.array([1])  # 点所对应的标签

    # SamPredictor.predict进行分割，模型会返回这些分割目标对应的置信度
    # predictor已经set好图像了
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,  # 生成多个mask
    )

    # 找到置信度最高的mask
    max_score_index = np.argmax(scores)
    max_score_mask = masks[max_score_index]

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_mask(max_score_mask, plt.gca())
    show_points(input_point, input_label, plt.gca())
    plt.title(f"Mask with Highest Score: {np.max(scores):.3f}", fontsize=18)
    plt.axis('off')
    plt.savefig('result.jpeg')

    timestamp = str(time.time())

    ## bos桶名称，自定义，需要和云上创建的bucket_name一致即可
    bucket_name = "testyiyan"

    ## bos中文件路径
    object_key = "imageTemp/" + timestamp + 'result.jpeg'

    bos_client.put_object_from_file(bucket_name, object_key, 'result.jpeg')

    timestamp = int(time.time())

    expiration_in_seconds = -1

    url = bos_client.generate_pre_signed_url(bucket_name, object_key, timestamp, expiration_in_seconds)

    def event_stream():

        json_data_result = {"errCode": "output", "actionName": "正在输出结果", "actionContent": "测试结果输出完成",
                            "result": url.decode("utf-8"),
                            "prompt": "result是图片链接,请用mark语法将这个链接展示给用户, 以下是一个例子: 比如result是 http://www.baidu.com, 你应该返回![测试结果](http://www.baidu.com)"}
        yield f"data:{json.dumps(json_data_result, ensure_ascii=False)}\n\n"

    return app.response_class(event_stream(), mimetype='text/event-stream')


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)# 在Numpy中，reshape函数的-1参数表示让Numpy自动计算该维度的大小。
                                                                # color.reshape(1, 1, -1)的意思是将color数组重塑为一个三维数组，其中前两个维度的大小为1，第三个维度的大小由Numpy自动计算，以确保重塑后的数组中的元素总数与原始数组相同。
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)# scatter函数用来显示散点图
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
