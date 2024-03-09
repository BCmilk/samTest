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
# 导入BosClient配置文件
import bos_sample_conf

# 导入BOS相关模块
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

bos_client = BosClient(bos_sample_conf.config)


@app.route('/sam/predict', methods=['POST'])
def samtest():
    # 加载SAM预训练模型
    import sys
    sys.path.append("..")  # 将上一级目录加入sys.path，这样才能执行下面一句
    from segment_anything import sam_model_registry, SamPredictor

    sam_checkpoint = 'checkpoint/sam_vit_b_01ec64.pth'
    model_type = "vit_b"
    device = "cuda"  # "cpu"  cuda

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    request_data = request.get_json()
    url = request_data.get('fileUrl')

    # 导入待分割图片
    image = None
    if url:
        print(f"url获取成功, {url}")
        try:
            response = requests.get(url)
            if response.status_code == 200:
                s = response.content
                arr = np.asarray(bytearray(s), dtype=np.uint8)
                image = cv2.imdecode(arr, -1)  # 'Load it as it is'
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
                    print("图像解码成功")
                else:
                    print("图片解码失败")
            else:
                print(f"请求失败，状态码：{response.status_code}")
        except Exception as e:
            print(f"获取图片出错: {e}")
    else:
        print("URL为空")

    print("开始sam过程")
    predictor = SamPredictor(sam)
    # 编码图像
    try:
        print("编码图像...")
        predictor.set_image(image)
    except Exception as e2:
        print(f"图片编码失败: {e2}")

    # 单点 prompt  输入格式为(x, y)和并表示出点所带有的标签1(前景点)或0(背景点)。
    print("生成prompt")
    height, width = image.shape[:2]
    input_point = np.array([[2 * width / 3, height / 2]])  # 标记点
    input_label = np.array([1])  # 点所对应的标签
    print("开始生成蒙版")
    # SamPredictor.predict进行分割，模型会返回这些分割目标对应的置信度
    # predictor已经set好图像了
    try:
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,  # 生成多个mask
        )
    except Exception:
        print("蒙版生成失败")

    # 找到置信度最高的mask
    max_score_index = np.argmax(scores)
    max_score_mask = masks[max_score_index]

    print("开始出图")
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_mask(max_score_mask, plt.gca())
    show_points(input_point, input_label, plt.gca())
    plt.title(f"Mask with Highest Score: {np.max(scores):.3f}", fontsize=18)
    plt.axis('off')
    plt.savefig('result.jpeg')

    print("出图完成，上传结果...")
    timestamp = str(time.time())

    ## bos桶名称，自定义，需要和云上创建的bucket_name一致即可
    bucket_name = "xxx"

    ## bos中文件路径
    object_key = "imageTemp/" + timestamp + 'result.jpeg'

    bos_client.put_object_from_file(bucket_name, object_key, 'result.jpeg')

    timestamp = int(time.time())

    expiration_in_seconds = -1
    print("上传成功，url如下: ")
    url = bos_client.generate_pre_signed_url(bucket_name, object_key, timestamp, expiration_in_seconds)
    print(url.decode("utf-8"))
    def event_stream():

        json_data_result = {"errCode": "output",
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
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)  # 在Numpy中，reshape函数的-1参数表示让Numpy自动计算该维度的大小。
    # color.reshape(1, 1, -1)的意思是将color数组重塑为一个三维数组，其中前两个维度的大小为1，第三个维度的大小由Numpy自动计算，以确保重塑后的数组中的元素总数与原始数组相同。
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)  # scatter函数用来显示散点图
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


@app.route("/.well-known/ai-plugin.json")
async def pluginManifest():
    host = request.host_url
    with open(".well-known/ai-plugin.json", encoding="utf-8") as f:
        text = f.read().replace("PLUGIN_HOST", host)
        return text, 200, {"Content-Type": "application/json"}


@app.route("/.well-known/openapi.yaml")
async def openapiSpec():
    host = request.host_url
    with open(".well-known/openapi.yaml", encoding="utf-8") as f:
        text = f.read().replace("PLUGIN_HOST", host)
        return text, 200, {"Content-Type": "text/yaml"}


@app.route("/.well-known/example.yaml")
async def exampleSpec():
    host = request.host_url
    with open(".well-known/example.yaml", encoding="utf-8") as f:
        text = f.read().replace("PLUGIN_HOST", host)
        return text, 200, {"Content-Type": "text/yaml"}


@app.route('/')
def index():
    return 'welcome to my webpage!'


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=8022)
