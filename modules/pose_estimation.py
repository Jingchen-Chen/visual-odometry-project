import cv2
import numpy as np

class PoseEstimator:
    def __init__(self):
        # Intrinsic Matrix K for KITTI Sequence 00
        self.K = np.array([[718.856, 0.0, 607.1928],
                          [0.0, 718.856, 185.2157],
                          [0.0, 0.0, 1.0]])

    def estimate_pose(self, kp1, kp2, matches):
        # Extract 2D coordinates of matched points
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

        # Compute Essential Matrix E
        E, mask = cv2.findEssentialMat(pts1, pts2, self.K, method=cv2.RANSAC, prob=0.999, threshold=0.5)

        # Recover pose (Rotation and Translation)
        _, R, t, mask = cv2.recoverPose(E, pts1, pts2, self.K, mask=mask)

        # 🔴 Core Correction: Matrix Inversion
        # OpenCV returns the transformation from 1 to 2. 
        # We need the pose of frame 2 relative to the coordinate system of frame 1.
        R_rel = R.T
        t_rel = -R.T @ t

        return R_rel, t_rel