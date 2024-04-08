import os
import cv2
import json
import time
import argparse
import requests
import numpy as np
from PIL import Image
from io import BytesIO

from flask import Flask, request
from flask import render_template

from urllib.parse import urlparse, unquote

from predict import predict

# -*- coding: utf-8 -*-
import oss2

# 填写RAM用户的访问密钥（AccessKey ID和AccessKey Secret）。
accessKeyId = 'your_acessKeyId'
accessKeySecret = 'your_accessKeySecret'
# 使用代码嵌入的RAM用户的访问密钥配置访问凭证。
auth = oss2.Auth(accessKeyId, accessKeySecret)

# 填写Bucket所在地域对应的Endpoint。以华东1（杭州）为例，Endpoint填写为https://oss-cn-hangzhou.aliyuncs.com。
# yourBucketName填写存储空间名称。
bucket = oss2.Bucket(auth, 'oss-cn-xxxxxxx.aliyuncs.com', 'yourBucketName')

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('demo01.html')


# 接收来自前端的本地图片路径
@app.route('/deploy_local', methods=['POST'])
def deploying_local():

    f = request.files['file']
    global filename
    filename = f.filename
    global file_path
    file_path = os.path.join("./static/" + filename)
    f.save(file_path)
    opt = parse_opt(file_path)
    detecting = predict(opt)
    result = detecting.main()
    output_path = "./static/output/" + str(filename)
    # print(output_path)
    # cv2.imwrite(output_path, result)
    result.save(output_path, exif=f.info.get('exif'))
    output_data = {'input_url': file_path, 'output_url': output_path}
    output_data = json.dumps(output_data)
    return output_data


# 接收来自服务后端的oss图片url
@app.route('/deploy_online', methods=['POST'])
def deploying_online():

    # 判断传入类型
    if request.content_type.startswith('application/json'):            
        input_url = request.json.get('input_url')
    elif request.content_type.startswith('multipart/form-data'):
        input_url = request.form.get('input_url')
    else:
        input_url = request.values.get("input_url")
    print("input_url:", input_url)

    # # 接收上传图片的url
    # input_url = "http://fufu-imgsubmit.oss-cn-beijing.aliyuncs.com/fuchuang%2Finput%2Fwell0_0088.jpg?OSSAccessKeyId=LTAI5tFoTQMdVcheJ1T1P8M8&Expires=1713098309&Signature=0AeJaGHeI5RcYIbKUj4qrKanmbI%3D"

    # 获取图片名称
    parsed_url = urlparse(input_url)
    relative_path = parsed_url.path
    file_name = relative_path.split('%2F')[-1]
    # print(file_name)

    # 获取图片numpy数据
    response = requests.get(input_url)
    global input_file
    input_file = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)

    # 保存原图片位置信息
    exif_data = Image.open(BytesIO(response.content)).info.get("exif")

    # 调用推理函数
    opt = parse_opt(input_file)
    img, box_info = predict(opt)
    
    start_post = time.time()

    # 设置输出路径
    output_path = "fuchuang/output/" + str(file_name)
    
    # 为推理结果添加位置信息
    img_pil = Image.fromarray(img)
    img_byte_array = BytesIO()
    if exif_data != None:
        img_pil.save(img_byte_array, format='JPEG', exif=exif_data)
    else:
        img_pil.save(img_byte_array, format='JPEG')
    img_code = img_byte_array.getvalue()

    # 将推理结果上传oss并获取url
    bucket.put_object(output_path, img_code)
    output_url = bucket.sign_url('GET', output_path, 5*60)
    print("output_url:", output_url)

    # 整理返回值
    output_data = {'input_url': input_url, 'output_url': output_url, 'box_info':str(box_info)}
    output_data = json.dumps(output_data)
    
    print('\n', output_data, '\n')
    end_post = time.time()
    print("后处理用时{}s".format(end_post - start_post))
    
    return output_data


# 批量接受来自服务后端的oss图片url
@app.route('/deploy_online_batch', methods=['POST'])
def deploying_online_batch():

    # 判断传入类型
    if request.content_type.startswith('application/json'):            
        input_url_list = request.json.get('input_url_list')
    elif request.content_type.startswith('multipart/form-data'):
        input_url_list = request.form.get('input_url_list')
    else:
        input_url_list = request.values.get("input_url_list")
    print("input_url_list:", input_url_list)

    # # 批量接收上传图片的url
    # input_url_list = \
    #     [
    #         "http://fufu-imgsubmit.oss-cn-beijing.aliyuncs.com/fuchuang%2Finput%2Fwell0_0088.jpg?OSSAccessKeyId=LTAI5tFoTQMdVcheJ1T1P8M8&Expires=1713098309&Signature=0AeJaGHeI5RcYIbKUj4qrKanmbI%3D",\
    #         "http://fufu-imgsubmit.oss-cn-beijing.aliyuncs.com/fuchuang%2Finput%2Fwell0_0088.jpg?OSSAccessKeyId=LTAI5tFoTQMdVcheJ1T1P8M8&Expires=1713098309&Signature=0AeJaGHeI5RcYIbKUj4qrKanmbI%3D", \
    #         "http://fufu-imgsubmit.oss-cn-beijing.aliyuncs.com/fuchuang%2Finput%2Fwell0_0088.jpg?OSSAccessKeyId=LTAI5tFoTQMdVcheJ1T1P8M8&Expires=1713098309&Signature=0AeJaGHeI5RcYIbKUj4qrKanmbI%3D"
    #     ]

    output_data_list = []
    for input_url in input_url_list:

        # 获取图片名称
        parsed_url = urlparse(input_url)
        relative_path = parsed_url.path
        file_name = relative_path.split('%2F')[-1]
        # print(file_name)

        # 获取图片numpy数据
        response = requests.get(input_url)
        global input_file
        input_file = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)

        # 保存原图片位置信息
        exif_data = Image.open(BytesIO(response.content)).info.get("exif")

        # 调用推理函数
        opt = parse_opt(input_file)
        img, box_info = predict(opt)

        # 设置输出路径
        output_path = "fuchuang/output/" + str(file_name)
        
        # 为推理结果添加位置信息
        img_pil = Image.fromarray(img)
        img_byte_array = BytesIO()
        if exif_data != None:
            img_pil.save(img_byte_array, format='JPEG', exif=exif_data)
        else:
            img_pil.save(img_byte_array, format='JPEG')
        img_code = img_byte_array.getvalue()

        # 将推理结果上传oss并获取url
        bucket.put_object(output_path, img_code)
        output_url = bucket.sign_url('GET', output_path, 5*60)
        print("output_url:", output_url)

        # 整理返回值
        output_data = {'input_url': input_url, 'output_url': output_url, 'box_info':str(box_info)}
        # output_data = json.dumps(output_data)
        output_data_list.append(output_data)
        
    print('\n', output_data_list, '\n')
        
    return output_data_list
    

# def parse_opt(input_file):
#     # 定义命令行参数
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--onnx_model", type=str, default="best.onnx", help="Input your ONNX model.")
#     parser.add_argument("--input_image", type=str, default=input_file, help="Input numpy image.")
#     parser.add_argument("--confidence_thres", type=float, default=0.5, help="Confidence threshold")
#     parser.add_argument("--iou_thres", type=float, default=0.5, help="NMS IoU threshold")
#     opt = parser.parse_args()
#     args = parser.parse_args(args=[])
#     return opt


def parse_opt(input_file) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, default="best.engine", help='Engine file')
    parser.add_argument('--imgs', type=str, default=input_file, help='Images file')
    # parser.add_argument('--show', type=bool, default=False, action='store_true', help='Show the detection results')
    # parser.add_argument('--out-dir', type=str, default='./output', help='Path to output file')
    parser.add_argument('--device', type=str, default='cuda:0', help='TensorRT infer device')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
    app.run(host='0.0.0.0', port=5000)

# if __name__ == '__main__':
#     start = time.time()
#     deploying_online_batch()
#     end = time.time()
#     print("总共用时{}s".format(end - start))  
