# -*- coding: utf-8 -*-
"""
    @Author  : YourZhou
    @Time    : 2019/11/4
    @Comment :
"""

import cv2 as cv

img = cv.imread("D:\\people_all\\ulite\\all_data\\full_img\\JPEGImages\\paper_2.jpg")
# 生成热力图
heat_img = cv.applyColorMap(img, cv.COLORMAP_JET)  # 注意此处的三通道热力图是cv2专有的GBR排列
output = cv.cvtColor(heat_img, cv.COLOR_BGR2RGB)  # 将BGR图像转为RGB图像

print(output.shape)
cv.imshow("123",output)
cv.waitKey(0)
cv.destroyAllWindows()