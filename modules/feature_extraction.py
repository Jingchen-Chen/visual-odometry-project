import cv2
import numpy as np

class FeatureExtractor:
    def __init__(self):
        # 初始化 ORB 检测器
        # nfeatures: 提取特征点的数量，2000-3000点对于VO来说足够了
        self.orb = cv2.ORB_create(nfeatures=3000)

    def extract_features(self, image):
        """
        输入: 灰度图像
        输出: keypoints (特征点坐标), descriptors (特征点描述子)
        """
        # 检测特征点并计算描述子
        keypoints, descriptors = self.orb.detectAndCompute(image, None)
        
        return keypoints, descriptors

    def draw_features(self, image, keypoints):
        """
        在图像上画出特征点，方便我们可视化调试
        """
        out_img = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0))
        return out_img

# 测试代码
if __name__ == "__main__":
    print("Feature Extractor Module Test")
    # 这里可以放一段简单的测试逻辑，读取一张图试试