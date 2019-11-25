# -*- coding: UTF-8 -*-
import os
from ulite.yolo_paper_infer.until import *
import time
import paddle.fluid as fluid
import codecs

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
label_dict = {}
with codecs.open('./label_list.txt') as f:
    for line in f:
        parts = line.strip().split()
        label_dict[str(float(parts[0]))] = parts[1]
print(label_dict)
class_dim = len(label_dict)

place = fluid.CPUPlace()
exe = fluid.Executor(place)
path = "./freeze_model3000"
[inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(dirname=path, executor=exe)
print(feed_target_names[0])

def draw_bbox_image(img, boxes, labels, save_name):
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
    if img.mode != 'RGB':
        img = img.convert('RGB')
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype('simsun.ttc', 50)
    for box, label in zip(boxes, labels):
        xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]
        draw.rectangle((xmin, ymin, xmax, ymax), None, 'red', 5)
        print(str(label))
        draw.text((xmin, ymin), label_dict[str(label)], (255, 255, 0), font=font)
    img.save(save_name)
    return img


def crop_bbox_image(img, boxes, labels, save_name):
    """
    将图片剪裁出矩形框
    :param img:
    :param boxes:
    :param save_name:
    :param labels
    :return:
    """
    # font = ImageFont.truetype("font.ttf", 25)
    # img_width, img_height = img.size
    if img.mode != 'RGB':
        img = img.convert('RGB')
    for box, label in zip(boxes, labels):
        xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]
        dt_img = img[ymin:ymax, xmin:xmax]
        dt_img.save(save_name)
    return img


def infer(image_path):
    """
    预测，将结果保存到一副新的图片中
    :param image_path:
    :return:
    """
    origin, tensor_img, resized_img = read_image(image_path)
    t1 = time.time()
    batch_outputs = exe.run(inference_program,
                            feed={feed_target_names[0]: tensor_img,},
                            fetch_list=fetch_targets)
    period = time.time() - t1
    print("predict cost time:{0}".format("%2.2f sec" % period))
    input_w, input_h = origin.size[0], origin.size[1]
    yolo_anchors, yolo_classes = get_yolo_anchors_classes(class_dim, anchors, anchor_mask)
    pred_boxes, pred_scores, pred_labels = get_all_yolo_pred(batch_outputs, yolo_anchors, yolo_classes,
                                                             (target_size[1], target_size[2]))
    boxes, scores, labels = calc_nms_box(pred_boxes, pred_scores, pred_labels, valid_thresh, nms_threshold)
    boxes = rescale_box_in_input_image(boxes, [input_h, input_w], target_size[1])
    print("result boxes: ", boxes)
    print("result scores:", scores)
    print("result labels:", labels)
    last_dot_index = image_path.rfind('.')
    out_path = image_path[:last_dot_index]
    out_path += '-reslut.jpg'
    draw_bbox_image(origin, boxes, labels, out_path)


if __name__ == '__main__':
    # image_path = sys.argv[1]
    name_list = os.listdir('D:/people_all/ulite/all_data/train_paper_data/test/')
    for name_ in name_list:
        image_path = os.path.join('D:/people_all/ulite/all_data/train_paper_data/test/', name_)
        infer(image_path)
