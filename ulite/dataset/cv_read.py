# -*- coding: utf-8 -*-
"""
    @Author  : YourZhou
    @Time    : 2019/11/4
    @Comment :
"""

import cv2 as cv

img = cv.imread("D:\\people_all\\ulite\\all_data\\train_paper_data\\test\\paper_5.jpg")
print(img.shape)
cv.putText(img, "P : " , (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
cv.imshow("123",img)
cv.waitKey(0)
cv.destroyAllWindows()