import cv2
import numpy as np


class FeatureExtractor:
    """
    改进版特征提取器：
    - 使用 SIFT 替代 ORB（尺度/旋转不变性更强，匹配更稳定）
    - 加入自适应直方图均衡化（CLAHE）预处理，提升低光/对比度差场景
    - 网格化均匀采样，防止特征点集中在纹理丰富区域导致估计偏差
    """

    def __init__(self, num_features: int = 2000, use_clahe: bool = True,
                 grid_rows: int = 4, grid_cols: int = 4):
        """
        Args:
            num_features: 期望提取的特征点总数
            use_clahe:    是否对图像做 CLAHE 预处理
            grid_rows/cols: 将图像分成几行几列的网格，每格均匀采样
        """
        self.num_features = num_features
        self.use_clahe = use_clahe
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols

        # SIFT：精度高，但比 ORB 慢；如需速度可改回 ORB
        self.sift = cv2.SIFT_create(nfeatures=num_features)

        # CLAHE 对比度受限的自适应直方图均衡化
        if use_clahe:
            self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    # ------------------------------------------------------------------
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """灰度图 CLAHE 预处理"""
        if self.use_clahe:
            return self.clahe.apply(image)
        return image

    # ------------------------------------------------------------------
    def extract_features(self, image: np.ndarray):
        """
        Returns:
            keypoints   : list of cv2.KeyPoint
            descriptors : np.ndarray (N, 128) float32
        """
        image = self._preprocess(image)
        h, w = image.shape[:2]

        cell_h = h // self.grid_rows
        cell_w = w // self.grid_cols
        kp_per_cell = max(1, self.num_features // (self.grid_rows * self.grid_cols))

        all_kp = []
        all_des = []

        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                # 裁出当前格子（带重叠边界防止边缘漏点）
                y1, y2 = r * cell_h, min((r + 1) * cell_h + 10, h)
                x1, x2 = c * cell_w, min((c + 1) * cell_w + 10, w)
                cell = image[y1:y2, x1:x2]

                # 在格子内检测
                sift_cell = cv2.SIFT_create(nfeatures=kp_per_cell)
                kps, des = sift_cell.detectAndCompute(cell, None)
                if kps is None or des is None:
                    continue

                # 将格子内坐标转回全图坐标
                for kp in kps:
                    kp.pt = (kp.pt[0] + x1, kp.pt[1] + y1)

                all_kp.extend(kps)
                all_des.append(des)

        if not all_kp:
            # Fallback：整图检测
            return self.sift.detectAndCompute(image, None)

        descriptors = np.vstack(all_des).astype(np.float32)
        return all_kp, descriptors

    # ------------------------------------------------------------------
    def draw_features(self, image: np.ndarray, keypoints) -> np.ndarray:
        return cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0))