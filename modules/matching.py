import cv2
import numpy as np


class FeatureMatcher:
    """
    改进版特征匹配器：
    - 使用 FLANN（近似最近邻）替代暴力匹配，速度更快
    - Lowe's Ratio Test (0.70 更严格)
    - 双向一致性验证（Cross-check），进一步去除误匹配
    """

    def __init__(self, ratio_thresh: float = 0.70):
        self.ratio_thresh = ratio_thresh

        # FLANN for SIFT (float descriptors)
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

    # ------------------------------------------------------------------
    def match_features(self, des1: np.ndarray, des2: np.ndarray):
        """
        Returns filtered good matches (list of cv2.DMatch).
        Falls back to BFMatcher if descriptors are too few.
        """
        if des1 is None or des2 is None:
            return []
        if len(des1) < 2 or len(des2) < 2:
            return []

        # 正向匹配
        matches_12 = self.flann.knnMatch(des1, des2, k=2)
        # 反向匹配（用于双向验证）
        matches_21 = self.flann.knnMatch(des2, des1, k=1)

        # 建立反向索引：des2的第i点 -> des1的第j点
        reverse_map = {}
        for m_list in matches_21:
            if m_list:
                reverse_map[m_list[0].queryIdx] = m_list[0].trainIdx

        good = []
        for m_pair in matches_12:
            if len(m_pair) < 2:
                continue
            m, n = m_pair
            # Ratio test
            if m.distance >= self.ratio_thresh * n.distance:
                continue
            # Cross-check：反向最近邻也指回 m.queryIdx
            if reverse_map.get(m.trainIdx) == m.queryIdx:
                good.append(m)

        return good

    # ------------------------------------------------------------------
    def draw_matches(self, img1, kp1, img2, kp2, matches, max_draw: int = 50):
        return cv2.drawMatches(
            img1, kp1, img2, kp2,
            matches[:max_draw], None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )