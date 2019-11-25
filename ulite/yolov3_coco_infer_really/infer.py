# -*- coding: utf-8 -*-
"""
    @Author  : YourZhou
    @Time    : 2019/11/5
    @Comment :
"""

import numpy as np
import time
import paddle.fluid as fluid
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from matplotlib import pyplot as plt
import os
import zipfile

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

label_dict = ['paper']
class_dim = 1
print("label_dict:{} class dim:{}".format(label_dict, class_dim))
place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
exe = fluid.Executor(place)
path = "./yolov3_15999"
[inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(dirname=path, executor=exe)


def draw_bbox_image(img, boxes, labels, scores, save_name):
    """
    给图片画上外接矩形框
    :param img:
    :param boxes:
    :param save_name:
    :param labels
    :return:
    """
    # font = ImageFont.truetype("font.ttf", 25)
    # img_width, img_height = img.size
    save_path = save_name + "/result.jpg"
    if img.mode != 'RGB':
        img = img.convert('RGB')
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype('simsun.ttc', 50)
    for box, label in zip(boxes, labels):
        xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]
        draw.rectangle((xmin, ymin, xmax, ymax), None, 'red', 5)
        draw.text((xmin, ymin), label_dict[int(label)] + str(scores), (255, 255, 0), font=font)
    img.save(save_path)
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
    result_path = save_path + '/crop_img.jpg'
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
    result_path = save_path + '/result_img.jpg'
    # font = ImageFont.truetype("font.ttf", 25)
    # img_width, img_height = img.size
    if img.mode != 'RGB':
        img = img.convert('RGB')
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype('simsun.ttc', 50)
    for box, label in zip(boxes, labels):
        xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]
        crop_paper(box, img, out_path)
        draw.rectangle((xmin, ymin, xmax, ymax), None, 'red', 5)
        # print(str(label))
        draw.text((xmin, ymin), "paper" + str(scores[0]), (255, 255, 0), font=font)
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


def infer(image_path):
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
    batch_outputs = exe.run(inference_program,
                            feed={feed_target_names[0]: tensor_img,
                                  feed_target_names[1]: image_shape[np.newaxis, :]},
                            fetch_list=fetch_targets,
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
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    draw_bbox_crop_image(origin, boxes, labels, scores, out_path)

    zipDir(out_path, out_path + ".zip")
    print(os.path.abspath(os.path.dirname(out_path)))


if __name__ == '__main__':
    # image_name = sys.argv[1]
    # image_path = image_name
    image_path = "D:/people_all/ulite/all_data/train_paper_data/test/paper_40.jpg"
    infer(image_path)
