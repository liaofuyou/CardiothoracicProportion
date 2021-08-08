import cv2
import matplotlib.pyplot as plt
import numpy as np


def find_max_area(contours):
    # 找到最大的轮廓
    area = []
    for k in range(len(contours)):
        area.append(cv2.contourArea(contours[k]))
    max_idx = np.argmax(np.array(area))
    return max_idx


def morph_open(threshold):
    # 开运算去除噪声
    kernel = np.ones((3, 3), np.uint8)
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel, iterations=3)
    return threshold


def get_left_right(cnt):
    # 计算最左  和最后的点, 外边距的x,y,w,h
    left = tuple(cnt[cnt[:, :, 0].argmin()][0])
    right = tuple(cnt[cnt[:, :, 0].argmax()][0])
    x, y, w, h = cv2.boundingRect(cnt)
    return left, right, x, y, w, h


def calc_diaphragm(image, hy, cx, cw, step):
    # 为了找出右膈肌顶的位置
    # 这种方法比较粗糙（希望有更好的方法请告诉我！！），通过大致找到右肺，和根据心脏的左侧，圈出一小块
    # 循环往下走，计算出现黑色像素就暂时，这个位置就右侧膈肌顶
    im = image.copy()
    width = int(cw / 8)
    while True:
        im = image[hy:hy + step, cx + width:cx + width * 2]
        no_black_pixels = cv2.countNonZero(im)
        if no_black_pixels < int(width * step * 0.95):
            return hy + step
        hy = hy + step


def compute_cardio_thoracic_proportion(image, segmentation, im_path):
    # 找出心脏位置
    ret, heart = cv2.threshold(segmentation, 1, 255, 0)
    heart = morph_open(heart)
    contours, hierarchy = cv2.findContours(heart, 1, 2)
    max_idx = find_max_area(contours)
    heart = contours[max_idx]
    # 得到心脏的最左的点和最右的点，和边界矩形的x，y，width,height
    left_heart, right_heart, hx, hy, width_heart, hh = get_left_right(heart)

    # 这次是找出肺部位置
    img_temp = segmentation.copy()
    w, h = img_temp.shape[0], img_temp.shape[1]
    # 把心脏和肺部当做一个整体，方便计算边界矩形
    ret, chest = cv2.threshold(img_temp, 0, 255, 0)
    chest = morph_open(chest)
    contours, hierarchy = cv2.findContours(chest, 1, 2)
    max_idx = find_max_area(contours)
    chest = contours[max_idx]
    # 得到肺部的最左的点和最右的点，和外边距的x，y，width,height
    left_chest, right_chest, cx, cy, width_chest, ch = get_left_right(chest)

    # 根据上面找到肺的最左位置和width传入找膈肌顶的函数中，
    # 缩小肺的范围，再找一次肺的最左的点和最右的点，和外边距的x，y，width,height
    position_y = calc_diaphragm(img_temp, left_heart[1], left_chest[0], width_chest, 5)
    ret, chest = cv2.threshold(img_temp[:position_y, :], 0, 255, 0)

    chest = morph_open(chest)
    contours, hierarchy = cv2.findContours(chest, 1, 2)
    max_idx = find_max_area(contours)
    chest = contours[max_idx]
    # 得到第二次肺部的最左的点和最右的点，和外边距的x，y，width,height
    left_chest, right_chest, cx, cy, width_chest, ch = get_left_right(chest)

    # 画出肺部的最大横径 和心脏的横径
    cv2.line(image, left_heart, (left_heart[0] + int(width_heart / 2), left_heart[1]), (0, 255, 0), 2)
    cv2.line(image, right_heart, (right_heart[0] - int(width_heart / 2), right_heart[1]), (0, 255, 0), 2)
    cv2.line(image, left_chest, (left_chest[0] + width_chest, left_chest[1]), (0, 0, 255), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX  # 定义字体

    # 计算心胸比
    cv2.putText(image, 'cardio-thoracic proportion:{}'.format(round(width_heart / width_chest, 2)), (10, 40), font, 0.8, (255, 255, 255), 2)

    #  显示
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.title("cardio-thoracic proportion:{}".format(round(width_heart / width_chest, 2)))
    plt.show()

    #  保存
    result_path = im_path.split('.')[0] + '_result.jpg'
    cv2.imwrite(result_path, image)
