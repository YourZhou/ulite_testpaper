# -*- coding: utf-8 -*-
"""
    @Author  : YourZhou
    @Time    : 2019/11/4
    @Comment :
"""

import cv2 as cv
import os
from xml.dom.minidom import parse
import xml.etree.ElementTree as ET
import numpy as np


def readXML():
    xml_path = "D:/people_all/ulite/all_data/full_img/Annotations/paper_1.xml"
    domTree = parse(xml_path)
    # 文档根元素
    rootNode = domTree.documentElement
    print(rootNode.nodeName)

    gt_boxs = rootNode.getElementsByTagName("object")
    print(gt_boxs[0].firstChild.data)
    print("****标注框信息****")
    for gt_box in gt_boxs:
        name = gt_box.getElementsByTagName("bndbox")[0]
        print(name.nodeName, ":", name.childNodes[0].data)
        if gt_box.hasAttribute("bndbox"):
            print("bndbox:{}".format(gt_box.hasAttribute("bndbox")))


def readXML2(xml_path, xml_name):
    # xml_path = "D:/people_all/ulite/all_data/full_img/Annotations/paper_1.xml"
    xml_name = str(xml_name.split('.')[0])
    tree = ET.parse(xml_path + xml_name + ".xml")
    root = tree.getroot()
    # for child in root:
    #     print(child.tag, child.attrib)

    gtbox = []
    """
    [0]:gt_box_xmin
    [1]:gt_box_ymin
    [2]:gt_box_xmax
    [3]:gt_box_ymax
    """
    gtbox.append(int(root[6][4][0].text))
    gtbox.append(int(root[6][4][1].text))
    gtbox.append(int(root[6][4][2].text))
    gtbox.append(int(root[6][4][3].text))

    # print(gtbox)
    return gtbox


def divide_paper(img_path, img_name, save_path, n=2, m=4):
    '''
    拆分图像
    :param img_path:
    :param img_name:
    :param save_path:
    n = 2  # 拆分多少行？
    m = 4  # 拆分多少列？
    :return:
    '''
    imgg = img_path + img_name
    img = cv.imread(imgg)
    #   img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    h = img.shape[0]
    w = img.shape[1]
    print('h={},w={},n={},m={}'.format(h, w, n, m))
    dis_h = int(np.floor(h / n))
    dis_w = int(np.floor(w / m))
    num = 0
    for i in range(n):
        for j in range(m):
            num += 1
            print('i,j={}{}'.format(i, j))
            sub = img[dis_h * i:dis_h * (i + 1), dis_w * j:dis_w * (j + 1), :]
            # save_imgs_path = save_path + '\\' + str(img_name.split('.')[0]) + '\\'
            # if not os.path.exists(save_imgs_path):
            #     os.makedirs(save_imgs_path)
            cv.imwrite(save_path + str(img_name.split('.')[0]) + '_{}.jpg'.format(num), sub)


def crop_paper(gt_paper, img_path, name, save_paper_path):

    # if not os.path.exists('output'):
    #     os.mkdir('output')

    full_img = cv.imread(img_path + name, 1)
    # 剪裁出真实paper坐标
    paper_img = full_img[gt_paper[1]:gt_paper[3], gt_paper[0]:gt_paper[2]]
    cv.imwrite(save_paper_path + name, paper_img)

    # paper_shape = paper_img.shape
    # print(paper_img.shape)


if __name__ == '__main__':
    img_path = "D:/people_all/ulite/all_data/worry_img/JPEGImages/"
    xml_path = "D:/people_all/ulite/all_data/worry_img/Annotations/"
    save_paper_path = 'D:/people_all/ulite/all_data/crop_img/crop_paper/'
    save_divide_paper = 'D:/people_all/ulite/all_data/crop_img/divide_paper/'

    # img_list = os.listdir(img_path)
    # for name in img_list:
    #     gt_paper = readXML2(xml_path, name)
    #     crop_paper(gt_paper, img_path, name, save_paper_path)

    # crop_imgs(paper_img, img_name, save_path)

    img_list = os.listdir(save_paper_path)
    for name in img_list:
        print(name)
        divide_paper(save_paper_path, name, save_divide_paper)
