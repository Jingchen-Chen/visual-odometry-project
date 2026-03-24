import cv2
import numpy as np

class FeatureExtractor:
    def __init__(self):
        # Initialize ORB detector
        # nfeatures: 2000-3000 points are typically sufficient for Visual Odometry (VO)
        self.orb = cv2.ORB_create(nfeatures=3000)

    def extract_features(self, image):
        """
        Input: Grayscale image
        Output: keypoints (coordinates), descriptors (feature vectors)
        """
        # Detect keypoints and compute descriptors
        keypoints, descriptors = self.orb.detectAndCompute(image, None)
        
        return keypoints, descriptors

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