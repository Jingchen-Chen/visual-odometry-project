import cv2

class FeatureMatcher:
    def __init__(self):
        # 对于 ORB 这种二进制描述子，使用 Hamming 距离是最快的
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def match_features(self, desc1, desc2):
        """
        输入: 两张图各自的描述子
        输出: 匹配结果
        """
        # 进行暴力匹配
        matches = self.bf.match(desc1, desc2)
        
        # 按照匹配距离排序（距离越短越匹配）
        matches = sorted(matches, key=lambda x: x.distance)
        
        return matches

    def draw_matches(self, img1, kp1, img2, kp2, matches):
        """
        画出前 N 个匹配点，检查是否有太多乱线（误匹配）
        """
        match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, 
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        return match_img