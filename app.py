import subprocess

from torch.backends.cuda import flash_sdp_enabled

import predict
from flask import Flask, render_template, request, jsonify
import os
from data_resize import crop_and_resize_image
import zipfile
import tempfile
import pandas as pd
from data_preprocessing import EyeDataset
from torchvision import transforms
from werkzeug.utils import secure_filename
import random
from test_model import evaluate
from markdown import markdown

from flask import request, jsonify
from datetime import datetime

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/chat')
def chat():
    return render_template('chat.html')


@app.route('/model')
def model():
    return render_template('model.html')


@app.route('/digital_doctor')
def digital_doctor():
    return render_template('digital_doctor.html')


@app.route('/doctor', methods=['GET'])
def doctor():
    # 设置运行 app.py 脚本所需的参数
    command = [
        'python', 'app.py', '--transport', 'webrtc', '--model', 'wav2lip', '--avatar_id', 'wav2lip_doctor'
    ]

    # 使用 subprocess 执行命令
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # 等待进程结束并捕获输出
    stdout, stderr = process.communicate()

    # 你可以在这里处理输出，或者作为响应的一部分返回
    if process.returncode == 0:
        response = {
            'status': 'success',
            'message': 'Wav2Lip 模型启动成功!',
            'output': stdout.decode('utf-8')
        }
    else:
        response = {
            'status': 'error',
            'message': '启动 Wav2Lip 模型失败。',
            'error': stderr.decode('utf-8')
        }

    # 返回响应给客户端
    return jsonify(response)


@app.route('/analysis')
def analysis():
    return render_template('analysis.html')


@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    user_message = data['message']
    think, answer = chatwithdeepseek(user_message)

    # 构建包含元数据的响应
    response_data = {
        'response': {
            'content': markdown(answer),
            'metadata': {
                'reasoning': think,  # 添加思考过程
                'timestamp': datetime.now().isoformat()
            }
        }
    }
    return jsonify(response_data)


# 上传并处理图片
@app.route('/upload', methods=['POST'])
def upload():
    if 'left_image' not in request.files or 'right_image' not in request.files:
        return jsonify({'error': 'No file part'})

    left_file = request.files['left_image']
    right_file = request.files['right_image']

    if left_file.filename == '' or right_file.filename == '':
        return jsonify({'error': 'No selected file'})

    if left_file and right_file:
        # 保存上传的文件
        left_img_path = os.path.join('static/images', left_file.filename)
        right_img_path = os.path.join('static/images', right_file.filename)

        # 将路径中的反斜杠转换为斜杠
        left_img_path = os.path.normpath(left_img_path)

        left_file.save(left_img_path)
        right_file.save(right_img_path)

        # 处理左右眼的照片,只保留为384*384的圆底图
        crop_and_resize_image(left_img_path, left_img_path)
        crop_and_resize_image(right_img_path, right_img_path)

        # 绘制Grad-CAM热力图
        test_grad_cam_dual(left_img_path, right_img_path)

        # 进行预测并显示结果
        result = predict.predict_main([left_img_path], [right_img_path])

        # 根据预测结果返回响应
        disease = ["正常", "糖尿病", "青光眼", "白内障", "AMD", "高血压", "近视", "其他疾病/异常"]
        diagnosis = disease[result]

        # suggestion=deepseek_response_result(message='',disease=diagnosis)
        suggestion = markdown(suggestion)
        print(f"deepseek思考内容{think}")

        print(left_img_path, right_img_path)

        dual_left_path = 'static\\result\\DualGradCAM_left.png'
        dual_right_path = 'static\\result\\DualGradCAM_right.png'

        return render_template('result.html', diagnosis=diagnosis,
                               left_image_path=left_img_path, right_image_path=right_img_path,
                               suggestion=suggestion, 
                               dual_left_path=dual_left_path, dual_right_path=dual_right_path)


# 批量上传诊断眼底图
@app.route('/batch_upload', methods=['POST', 'GET'])
def batch_upload():
    # 检查上传的文件
    if 'image_zip' not in request.files or 'description_excel' not in request.files:
        return jsonify({'error': '缺少上传的文件'})

    # 上传的文件
    image_zip_file = request.files['image_zip']  # 包含图像的压缩包
    excel_file = request.files['description_excel']  # 数据集的Excel文件

    if image_zip_file.filename == '' or excel_file.filename == '':
        return jsonify({'error': '文件未选择'})

    # 保存上传的Excel文件
    excel_filename = secure_filename(excel_file.filename)
    excel_path = os.path.join('static', 'uploads', excel_filename)
    os.makedirs(os.path.dirname(excel_path), exist_ok=True)
    excel_file.save(excel_path)

    # 解压图像压缩包
    zip_filename = secure_filename(image_zip_file.filename)
    zip_path = os.path.join('static', 'uploads', zip_filename)
    image_zip_file.save(zip_path)
    extract_dir = os.path.join('static', 'uploads', os.path.splitext(zip_filename)[0])
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    # 图像预处理
    base_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # 加载数据集
    dataset = EyeDataset(dataset_dir=extract_dir, label_file=excel_path, transform=base_transform)

    predictions = []
    ground_truths = []
    labels = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
    query_labels = {'N': '正常', 'D': '糖尿病', 'G': '青光眼', 'C': '白内障', 'A': 'AMD', 'H': '高血压', 'M': '近视',
                    'O': '其他疾病/异常'}

    for i in range(len(dataset)):
        # 获取图像路径
        left_img_path = os.path.join(extract_dir, image_zip_file.filename.split('.')[0],
                                     dataset.labels.iloc[i]['Left-Fundus'])
        right_img_path = os.path.join(extract_dir, image_zip_file.filename.split('.')[0],
                                      dataset.labels.iloc[i]['Right-Fundus'])


        # 打印图像路径，以调试路径是否正确
        print(f"Patient {i}: Left image path: {left_img_path}, Right image path: {right_img_path}")

        # 检查图像路径是否存在
        if not os.path.exists(left_img_path) or not os.path.exists(right_img_path):
            print(f"Skipping patient {i} due to missing images")
            continue

        # 处理左右眼的照片,只保留为384*384的圆底
        crop_and_resize_image(left_img_path, left_img_path)
        crop_and_resize_image(right_img_path, right_img_path)

        # 进行预测
        result = predict.predict_main([left_img_path], [right_img_path])
        # print(f"Prediction for patient {i}: {labels[result]}")
        # predictions.append(labels[result])
        # 选择指定行和列，并转换为数值
        values = pd.to_numeric(dataset.labels.iloc[i][['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']], errors='coerce')
        # 直接返回最大值对应的标签
        labels_with_ones = values[values == 1].index.tolist()
        print(f"Prediction for patient  {i}: {labels_with_ones}")
        print(f"Ground truth for patient {i}: {labels_with_ones}")
        ground_truths.append(labels_with_ones)

    print(f"Length of predictions: {len(predictions)}")
    print(f"Length of ground_truths: {len(ground_truths)}")
    # 计算准确率
    total = len(predictions)
    correct = sum([1 for p, t in zip(predictions, ground_truths) if p == t])
    accuracy_person = correct / total if total > 0 else 0.0

    # 预测模型评价
    accuracy_disease, precision, recall, f1, conf_matrix_df = evaluate(predictions, ground_truths)
    roc_curve_path = 'static\\roc_curve.png'
    confusion_matrix_heatmap_path = 'static\\confusion_matrix_heatmap.png'
    pr_curve_path = 'static\\result\\pr_curve.png'
    print(roc_curve_path)
    print(confusion_matrix_heatmap_path)

    # 传递批量诊断结果到模板
    return render_template('batch_result.html', accuracy_person=accuracy_person, accuracy_disease=accuracy_disease,
                           predictions=predictions,
                           ground_truths=ground_truths, total=total, query_labels=query_labels, precision=precision,
                           recall=recall, f1=f1, conf_matrix_df=conf_matrix_df,
                           roc_curve_path=roc_curve_path, confusion_matrix_heatmap_path=confusion_matrix_heatmap_path,
                           pr_curve_path=pr_curve_path)


if __name__ == '__main__':
    app.run(debug=False)
