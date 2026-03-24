import cv2

class FeatureMatcher:
    def __init__(self):
        # 注意：使用 KNN 匹配时，crossCheck 必须为 False
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def match_features(self, desc1, desc2):
        """
        使用 KNN 和 Lowe's Ratio Test 进行高质量特征匹配
        """
        # 寻找每个特征点最近的 2 个匹配点 (k=2)
        matches = self.bf.knnMatch(desc1, desc2, k=2)
        
        good_matches = []
        # Lowe's Ratio Test
        for m, n in matches:
            # 如果第一近的距离 小于 第二近距离的 75% (0.75 是经验法则)
            # 说明这个匹配非常具有辨识度，不是容易混淆的重复纹理
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        
        return good_matches

    def draw_matches(self, img1, kp1, img2, kp2, matches):
        match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, 
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        return match_img