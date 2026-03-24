import numpy as np
import matplotlib.pyplot as plt


class VOPlotter:
    def __init__(self, gt_path):
        # 1. Load KITTI Ground Truth trajectory
        # KITTI GT format: N rows x 12 columns, each row is a flattened 3x4 transformation matrix (R|t)
        self.gt_poses = np.loadtxt(gt_path)
        self.num_frames = len(self.gt_poses)

        # Extract (x, z) coordinates for plotting (In KITTI, the motion plane is X-Z)
        self.gt_x = self.gt_poses[:, 3]
        self.gt_z = self.gt_poses[:, 11]

        # Store GT frame-0 pose for trajectory alignment
        self.gt_start = parse_gt_pose(self.gt_poses[0])

        # 2. Initialize estimated trajectory.
        # We do NOT pre-insert (0,0) here. add_estimated_pose() aligns each pose
        # into the GT world frame via gt_start, so the first appended point
        # will land correctly at the GT origin automatically.
        self.est_x = []
        self.est_z = []

    def add_estimated_pose(self, T_global):
        """
        Receives the accumulated camera pose T_global (4x4) expressed in the
        initial camera frame, and converts it into the KITTI world frame for plotting.

        T_global encodes: how the camera has moved since frame 0, in camera-0 coordinates.
        gt_start encodes: the world pose of camera-0.

        T_world = gt_start @ T_global  maps T_global into world coordinates,
        so est_x/est_z align directly with gt_x/gt_z.
        """
        T_world = self.gt_start @ T_global
        self.est_x.append(T_world[0, 3])
        self.est_z.append(T_world[2, 3])

    def save_trajectory_plot(self, save_path):
        """
        Plot the trajectory comparison and save it to the results/ directory.
        """
        plt.figure(figsize=(10, 8))

        # Ground Truth (blue)
        plt.plot(self.gt_x, self.gt_z, label='Ground Truth', color='blue', linewidth=1)

        # Estimated trajectory (red dashed).
        # est has num_frames-1 points (one per processed frame pair); gt has num_frames points.
        # We align against gt[1:] by skipping gt[0], or simply overlay — both start at the same world point.
        plt.plot(self.est_x, self.est_z,
                 label='Estimated VO (Scale Corrected)',
                 color='red', linestyle='--', linewidth=2)

        plt.title('Visual Odometry on KITTI Sequence 00', fontsize=15)
        plt.xlabel('X (meters)', fontsize=12)
        plt.ylabel('Z (meters)', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True)

        plt.savefig(save_path)
        plt.close()
        print(f"✅ Trajectory plot successfully saved to: {save_path}")


def parse_gt_pose(gt_frame):
    """
    Converts a row of 12 numbers (flattened 3x4 matrix) into a 4x4 homogeneous
    transformation matrix.
    """
    T = np.eye(4)
    T[:3, :4] = gt_frame.reshape(3, 4)
    return T