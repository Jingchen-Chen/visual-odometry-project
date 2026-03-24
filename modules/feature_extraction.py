import cv2
import numpy as np

class FeatureExtractor:
    def __init__(self):
        # GFTT finds corners robustly; ORB describes them
        self.detector = cv2.GFTTDetector_create(
            maxCorners=2000, qualityLevel=0.01,
            minDistance=10, blockSize=3)
        self.descriptor = cv2.ORB_create(nfeatures=2000)

    def extract_features(self, image):
        kp = self.detector.detect(image, None)
        kp, des = self.descriptor.compute(image, kp)
        return kp, des

    def draw_features(self, image, keypoints):
        """
        Draw keypoints on the image for visualization and debugging
        """
        out_img = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0))
        return out_img

# Test Code
if __name__ == "__main__":
    print("Feature Extractor Module Test")
    # Simple test logic: load an image and try extracting features here