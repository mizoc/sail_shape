# -*- coding:utf-8 -*-
import sys

import cv2
import numpy as np


class SailShapeAnalyzer:
    def __init__(self):
        self.wname = "Sail Shape"
        self.point_list = []  # マストトップ、グースネック、ツイスト位置にする。
        self.depth_line = []  # 25,50,75%のラインを引く際に使う、それぞれの端っこの座標。
        self.quarter_point_list = []  # マスト上の25,50,75%の座標
        self.display_result = []  # 左上に表示する結果
        self.point_num = 6
        self.mast_len = 1
        self.twist_len = 0
        self.twist_loc = 0
        self.calculate_line_length = lambda dots: np.sqrt((dots[0][0] - dots[1][0]) ** 2 + (dots[0][1] - dots[1][1]) ** 2)

    def onMouse(self, event, x, y, flag, raw_img):
        img = raw_img.copy()

        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.point_list) < self.point_num:
                self.point_list.append([x, y])

        # レーダーの描画
        if len(self.point_list) < self.point_num:
            h, w = img.shape[0], img.shape[1]
            cv2.line(img, (x, 0), (x, h), (255, 0, 0), 1, cv2.LINE_AA)
            cv2.line(img, (0, y), (w, y), (255, 0, 0), 1, cv2.LINE_AA)

        # マストの線の描画
        if 1 < len(self.point_list):
            cv2.line(img, (self.point_list[1][0], self.point_list[1][1]), (self.point_list[0][0], self.point_list[0][1]), (0, 255, 0), 2, cv2.LINE_AA)
        elif 1 == len(self.point_list):
            cv2.line(img, (x, y), (self.point_list[0][0], self.point_list[0][1]), (0, 255, 0), 2, cv2.LINE_AA)

        if len(self.point_list) == 3 and len(self.display_result) == 0:
            # 垂線の足を計算
            a = self.point_list[1][1] - self.point_list[0][1]
            b = self.point_list[0][0] - self.point_list[1][0]
            c = self.point_list[0][1] * self.point_list[1][0] - self.point_list[0][0] * self.point_list[1][1]
            self.px = int((self.point_list[2][0] * (b**2) - a * b * self.point_list[2][1] - a * c) / (a**2 + b**2))
            self.py = int((self.point_list[2][1] * (a**2) - a * b * self.point_list[2][0] - b * c) / (a**2 + b**2))
            # self.point_list.append([self.px, self.py])

            self.mast_len = self.calculate_line_length(self.point_list[0:2])  # マストの長さ
            self.twist_loc = self.calculate_line_length([[self.px, self.py], self.point_list[0]]) / self.mast_len  # ツイスト位置。マストトップからマスト長の何％の位置にあるか。
            self.twist_len = self.calculate_line_length([[self.px, self.py], self.point_list[2]]) / self.mast_len  # ツイストの量。ツイスト位置からマストに下ろした垂線の長さ。

            self.display_result.append(f"Twist Location: {np.round(self.twist_loc*100, decimals=2)}%")
            self.display_result.append(f"Twist Amount: {np.round(self.twist_len*100, decimals=2)}%")

            # 25%の点を出す
            theta = np.arctan(-a / b)
            for i in range(1, 4):
                x25 = self.point_list[0][0] + np.sign(theta) * (self.mast_len * np.cos(theta) * i / 4)
                y25 = self.point_list[0][1] + np.sign(theta) * (self.mast_len * np.sin(theta) * i / 4)
                d = b * x25 - a * y25  # -bx+ay+d=0

                self.quarter_point_list.append([x25, y25])
                self.depth_line.append([[0, int(-d / a)], [img.shape[1], int((-d + b * (img.shape[1])) / a)]])
            # print(x25, y25)
            # print(theta)
            # print(int(-d / a))
            # print(img.shape[1], int((-d + b * (img.shape[1])) / a))

        if len(self.point_list) > 2:
            cv2.line(img, (self.px, self.py), (self.point_list[2][0], self.point_list[2][1]), (0, 255, 0), 2, cv2.LINE_AA)
            for i in self.depth_line:
                cv2.line(img, (i[0][0], i[0][1]), (i[1][0], i[1][1]), (0, 255, 0), 2, cv2.LINE_AA)

        if len(self.point_list) == 6:
            self.sail_depth = [self.calculate_line_length([self.quarter_point_list[i], self.point_list[3 + i]]) / self.mast_len for i in range(3)]
            for i in range(3):
                self.display_result.append(f"{(i+1)*25}% Depth: {np.round(self.sail_depth[i]*100, decimals=2)}%")
            self.point_list.append([-10, -10])  # Dammy

        if len(self.point_list) > 6:
            for i, text in enumerate(self.display_result, 1):
                cv2.putText(
                    img,
                    text,
                    (5, 20 * i),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )

        # 点の描画
        for i in range(len(self.point_list)):
            cv2.circle(img, (self.point_list[i][0], self.point_list[i][1]), 3, (0, 0, 255), 3)

        cv2.imshow(self.wname, img)

    def scale_to_height(self, img, height):
        """高さが指定した値になるように、アスペクト比を固定して、リサイズする。"""
        h, w = img.shape[:2]
        width = round(w * (height / h))
        dst = cv2.resize(img, dsize=(width, height))

        return dst

    def run(self):
        # img = self.scale_to_height(cv2.imread("./close1.jpg"), 900)
        img = self.scale_to_height(cv2.imread(sys.argv[1]), 900)

        cv2.namedWindow(self.wname)
        cv2.setMouseCallback(self.wname, self.onMouse, img)
        cv2.imshow(self.wname, img)

        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    sail = SailShapeAnalyzer()
    sail.run()
