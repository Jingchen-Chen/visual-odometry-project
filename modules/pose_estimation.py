import cv2
import numpy as np

class PoseEstimator:
    def __init__(self):
        # Intrinsic Matrix K for KITTI Sequence 00
        self.K = np.array([[718.856, 0.0, 607.1928],
                          [0.0, 718.856, 185.2157],
                          [0.0, 0.0, 1.0]])

    def estimate_pose(self, kp1, kp2, matches):
        if len(matches) < 30:
            return None, None  # caller must handle this

        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

        E, mask = cv2.findEssentialMat(pts1, pts2, self.K,
                                       method=cv2.RANSAC, prob=0.999, threshold=0.5)
        if E is None:
            return None, None

        inliers, R, t, mask = cv2.recoverPose(E, pts1, pts2, self.K, mask=mask)
        if inliers < 20:
            return None, None  # too few inliers, skip frame

        # cv2.recoverPose returns R, t such that:
        #   X_cam2 = R @ X_cam1 + t
        # This is the transform FROM world TO camera2 (i.e. the camera moved).
        # For trajectory integration we need the camera's motion in world coordinates,
        # i.e. the pose of camera2 expressed in camera1's frame:
        #   P_world2 = P_world1 + R1 * t_rel  →  t_rel = R.T @ (-t), R_rel = R.T
        # This converts from "how the world moved relative to camera" to
        # "how the camera moved relative to the world".
        R_rel = R.T
        t_rel = -R.T @ t

        return R_rel, t_rel