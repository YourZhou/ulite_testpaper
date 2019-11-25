# -*- coding: utf-8 -*-
"""
    @Author  : YourZhou
    @Time    : 2019/11/4
    @Comment :
"""
import os

# path = input('请输入文件路径(结尾加上/)：')
path = 'D:/people_all/ulite/all_data/acc_img/JPEGImages/'
# 获取该目录下所有文件，存入列表中
f = os.listdir(path)
a = 1
n = 21
for i in f:
    # 设置旧文件名（就是路径+文件名）
    oldname = path + i
    # oldname = path + 'IMG_' + str(a) + '.jpg'

    # 设置新文件名
    newname = path + 'paper_' + str(n) + '.jpg'

    # 用os模块中的rename方法对文件改名
    os.rename(oldname, newname)
    print(oldname, '======>', newname)
    a += 1
    n += 1
