# -*- coding: utf-8 -*-
"""
    @Author  : YourZhou
    @Time    : 2019/11/5
    @Comment :
"""

import numpy as np
import time
import json
import paddle.fluid as fluid
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from matplotlib import pyplot as plt
from flask import Flask, request, send_from_directory, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import datetime
import zipfile
import os

app = Flask(__name__)

use_gpu = True
ues_tiny = False

yolo_config = {
    "input_size": [3, 608, 608],
    "anchors": [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326],
    "anchor_mask": [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
}

target_size = yolo_config['input_size']
anchors = yolo_config['anchors']
anchor_mask = yolo_config['anchor_mask']

nms_threshold = 0.4
valid_thresh = 0.4
confs_threshold = 0.5

"""
label_dict：模型类别
class_dim：标签数
worry_data:瑕疵统计
worry_num:创建数组进行瑕疵计数
"""
label_dict1 = ['paper']
label_dict2 = ['Crooked', 'Brief', 'Stain', 'Burrs', 'Sign', 'Fold']

worry_num = []
for i in range(6):
    worry_num.append(0)
print(worry_num)
class_dim1 = 1
class_dim2 = 6
print("label_dict1:{} class dim1:{}".format(label_dict1, class_dim1))
print("label_dict2:{} class dim2:{}".format(label_dict2, class_dim2))

"""模型读取"""
place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
exe = fluid.Executor(place)
path1 = "./yolov3_15999"
path2 = "./yolov3_165999"
[inference_program1, feed_target_names1, fetch_targets1] = fluid.io.load_inference_model(dirname=path1, executor=exe)

scope = fluid.Scope()
with fluid.scope_guard(scope):
    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    [inference_program2, feed_target_names2, fetch_targets2] = (
        fluid.io.load_inference_model(dirname=path2, executor=exe))


def draw_boxes_on_image(image_path,
                        boxes,
                        scores,
                        labels,
                        label_names,
                        score_thresh=0.5):
    image = np.array(Image.open(image_path))
    plt.figure()
    _, ax = plt.subplots(1)
    ax.imshow(image)

    image_name = image_path.split('/')[-1]
    print("Image {} detect: ".format(image_name))
    colors = {}
    for box, score, label in zip(boxes, scores, labels):
        if score < score_thresh:
            continue
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        label = int(label)
        if label not in colors:
            colors[label] = plt.get_cmap('hsv')(label / len(label_names))
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
        rect = plt.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            linewidth=2.0,
            edgecolor=colors[label])
        ax.add_patch(rect)
        ax.text(
            x1,
            y1,
            '{} {:.4f}'.format(label_names[label], score),
            verticalalignment='bottom',
            horizontalalignment='left',
            bbox={'facecolor': colors[label],
                  'alpha': 0.5,
                  'pad': 0},
            fontsize=8,
            color='white')
        print("\t {:15s} at {:25} score: {:.5f}".format(label_names[int(
            label)], str(list(map(int, list(box)))), score))
    image_name = image_name.replace('jpg', 'png')
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(
        "./output/{}".format(image_name), bbox_inches='tight', pad_inches=0.0)
    print("Detect result save at ./output/{}\n".format(image_name))
    plt.cla()
    plt.close('all')


def draw_bbox_image(img, boxes, labels, scores, save_path, num):
    """
    给图片画上外接矩形框
    :param img:
    :param boxes:
    :param save_name:
    :param labels
    :return:
    """
    save_name = save_path + "/worry_img_" + str(num) + ".jpg"
    # font = ImageFont.truetype("font.ttf", 25)
    # img_width, img_height = img.size
    if img.mode != 'RGB':
        img = img.convert('RGB')
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype('simsun.ttc', 25)
    for box, label, score in zip(boxes, labels, scores):
        if (score >= 0.7):
            # 存入瑕疵计数数组
            worry_num[int(label)] += 1
            xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]
            draw.rectangle((xmin, ymin, xmax, ymax), None, 'red', 3)
            # goal = " {:.2f}%".format(score * 100)
            draw.text((xmin, ymin), label_dict2[int(label)], (0, 255, 0), font=font)
            img.save(save_name)
    return img


def divide_paper(img, out_path, n=2, m=4):
    '''
    拆分图像
    :param img_path:
    :param img_name:
    :param out_path:
    :param n = 2  # 拆分多少行？
    :param m = 4  # 拆分多少列？
    :return:
    '''
    save_path = out_path + "/divide"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    result_path = save_path + '/divide_img'
    #   img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    h = img.size[1]
    w = img.size[0]
    print('real paper h={},w={},n={},m={}'.format(h, w, n, m))
    dis_h = int(np.floor(h / n))
    dis_w = int(np.floor(w / m))
    num = 0
    for i in range(n):
        for j in range(m):
            num += 1
            # print('i,j={}{}'.format(i, j))
            # sub = img[dis_h * i:dis_h * (i + 1), dis_w * j:dis_w * (j + 1), :]
            divide_img = img.crop((dis_w * j, dis_h * i, dis_w * (j + 1), dis_h * (i + 1)))

            # cv.imwrite(save_path + str(img_name.split('.')[0]) + '_{}.jpg'.format(num), sub)
            divide_img.save(result_path + '_{}.jpg'.format(num))


def crop_paper(gt_paper, full_img, out_path):
    '''
    提取检测区域图像
    :param gt_paper:
    :param full_img:
    :param out_path:
    :return:
    '''

    save_path = out_path + "/crop"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    result_path = save_path + '/crop_img_1.jpg'
    # 剪裁出真实paper坐标
    # paper_img = full_img[gt_paper[1]:gt_paper[3], gt_paper[0]:gt_paper[2]]
    paper_img = full_img.crop((gt_paper[0], gt_paper[1], gt_paper[2], gt_paper[3]))
    paper_img.save(result_path)
    divide_paper(paper_img, out_path)
    # print(os.path.abspath(os.path.dirname(result_path)))


def draw_bbox_crop_image(img, boxes, labels, scores, out_path):
    """
    给图片画上外接矩形框并截取
    :param img:
    :param boxes:
    :param save_name:
    :param labels
    :return:
    """
    save_path = out_path + "/result"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    result_path = save_path + '/result_img_1.jpg'
    # font = ImageFont.truetype("font.ttf", 25)
    # img_width, img_height = img.size
    if img.mode != 'RGB':
        img = img.convert('RGB')
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype('simsun.ttc', 50)
    for box, label, score in zip(boxes, labels, scores):
        xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]
        crop_paper(box, img, out_path)
        draw.rectangle((xmin, ymin, xmax, ymax), None, 'red', 5)
        goal = " {:.2f}%".format(score * 100)
        # print(str(label))
        draw.text((xmin, ymin), label_dict1[int(label)] + goal, (0, 255, 0), font=font)
        break
    img.save(result_path)
    return img


def resize_img(img, target_size):
    """
    保持比例的缩放图片
    :param img:
    :param target_size:
    :return:
    """
    img = img.resize(target_size[1:], Image.BILINEAR)
    return img


def read_image(img_path):
    """
    读取图片
    :param img_path:
    :return:
    """
    origin = Image.open(img_path)
    img = resize_img(origin, target_size)
    resized_img = img.copy()
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = np.array(img).astype('float32').transpose((2, 0, 1))  # HWC to CHW
    img -= 127.5
    img *= 0.007843
    img = img[np.newaxis, :]
    return origin, img, resized_img


def zipDir(dirpath, outFullName):
    """
    压缩指定文件夹
    :param dirpath: 目标文件夹路径
    :param outFullName: 压缩文件保存路径+xxxx.zip
    :return: 无
    """
    zip = zipfile.ZipFile(outFullName, "w", zipfile.ZIP_DEFLATED)
    for path, dirnames, filenames in os.walk(dirpath):
        # 去掉目标跟路径，只对目标文件夹下边的文件及文件夹进行压缩
        fpath = path.replace(dirpath, '')

        for filename in filenames:
            zip.write(os.path.join(path, filename), os.path.join(fpath, filename))
    zip.close()


def worry2json(label, json_path):
    """将错误信息保存为json格式文件"""
    all_num = 0
    for ele in range(0, len(label)):
        all_num = all_num + label[ele]

    worry_json = {
        'Crooked': label[0],
        'Brief': label[1],
        'Stain': label[2],
        'Burrs': label[3],
        'Sign': label[4],
        'Fold': label[5],
        'ALL': all_num
    }
    json_str = json.dumps(worry_json)
    data = json.loads(json_str)
    # Writing JSON data
    with open(json_path, 'w') as f:
        json.dump(data, f)

    return json_str


def infer2(image_path, out_path, num):
    """
    预测，将结果保存到一副新的图片中
    :param image_path:
    :return:
    """
    origin, tensor_img, resized_img = read_image(image_path)
    input_w, input_h = origin.size[0], origin.size[1]
    image_shape = np.array([input_h, input_w], dtype='int32')
    # print("image shape high:{0}, width:{1}".format(input_h, input_w))
    t1 = time.time()
    with fluid.scope_guard(scope):
        batch_outputs = exe.run(inference_program2,
                                feed={feed_target_names2[0]: tensor_img,
                                      feed_target_names2[1]: image_shape[np.newaxis, :]},
                                fetch_list=fetch_targets2,
                                return_numpy=False)
    period = time.time() - t1
    print("predict cost time:{0}".format("%2.2f sec" % period))
    bboxes = np.array(batch_outputs[0])
    # print(bboxes)

    if bboxes.shape[1] != 6:
        print("No object found in {}".format(image_path))
        return

    labels = bboxes[:, 0].astype('int32')
    print(labels)
    scores = bboxes[:, 1].astype('float32')
    print(scores)
    boxes = bboxes[:, 2:].astype('float32')

    save_path = out_path + "/worry"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    draw_bbox_image(origin, boxes, labels, scores, save_path, num)


@app.route('/all_infer', methods=['POST'])
def all_infer():
    print(request.files)
    f = request.files['img']

    # 保存图片
    save_father_path = 'images'
    dt = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    image_name = dt + '.' + secure_filename(f.filename).split('.')[-1]
    print(image_name)
    image_path = os.path.join(save_father_path, image_name)
    if not os.path.exists(save_father_path):
        os.makedirs(save_father_path)
    f.save(image_path)

    origin, tensor_img, resized_img = read_image(image_path)
    input_w, input_h = origin.size[0], origin.size[1]
    image_shape = np.array([input_h, input_w], dtype='int32')
    # print("image shape high:{0}, width:{1}".format(input_h, input_w))
    t1 = time.time()
    batch_outputs = exe.run(inference_program1,
                            feed={feed_target_names1[0]: tensor_img,
                                  feed_target_names1[1]: image_shape[np.newaxis, :]},
                            fetch_list=fetch_targets1,
                            return_numpy=False)
    period = time.time() - t1
    print("predict cost time:{0}".format("%2.2f sec" % period))
    bboxes = np.array(batch_outputs[0])
    # print(bboxes)

    if bboxes.shape[1] != 6:
        print("No object found in {}".format(image_path))
        return

    labels = bboxes[:, 0].astype('int32')
    print(labels)
    scores = bboxes[:, 1].astype('float32')
    boxes = bboxes[:, 2:].astype('float32')

    last_dot_index = image_path.rfind('.')
    out_path = image_path[:last_dot_index]
    out_path = out_path + "_out"

    # last_dot_index = image_name.rfind('.')
    # buf_name = image_name[:last_dot_index]
    # image_name = buf_name + "_out"

    if not os.path.exists(out_path):
        os.mkdir(out_path)
    draw_bbox_crop_image(origin, boxes, labels, scores, out_path)

    divide_path = out_path + "/divide/"
    img_list = os.listdir(divide_path)
    n = 1
    for name in img_list:
        print(name)
        worry_name = divide_path + name
        infer2(worry_name, out_path, n)
        n += 1

    json_name = out_path + "/data.json"
    json_str = worry2json(worry_num, json_name)
    print(json_str)
    # print(worry_num)
    for i in range(6):
        worry_num[i] = 0
    # print(worry_num)

    zip_name = out_path + ".zip"
    zipDir(out_path, zip_name)
    download_file = send_file(zip_name, as_attachment=True)
    return download_file


if __name__ == '__main__':
    # image_name = sys.argv[1]
    # image_path = image_name
    # image_path = "D:/people_all/ulite/all_data/train_paper_data/test/paper_40.jpg"
    # infer(image_path)
    app.run(host="192.168.43.171", port=8787)
