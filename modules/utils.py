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

        # 2. Initialize lists to store estimated trajectory
        self.est_x = [0.0]
        self.est_z = [0.0]

    def add_estimated_pose(self, T_global):
        """
        Receives current global transformation matrix (4x4) and extracts (x, z) translation
        """
        self.est_x.append(T_global[0, 3])
        self.est_z.append(T_global[2, 3])

    def save_trajectory_plot(self, save_path):
        """
        Plot the trajectory comparison and save it to the results/ directory
        """
        plt.figure(figsize=(10, 8))
        
        # Plot Ground Truth (Blue line)
        plt.plot(self.gt_x, self.gt_z, label='Ground Truth', color='blue', linewidth=1)
        
        # Plot Estimated Trajectory (Red dashed line)
        # Note: We slice the list if there's a frame mismatch (e.g., estimation starts from frame 2)
        plt.plot(self.est_x, self.est_z, label='Estimated VO (Scale Corrected)', 
                 color='red', linestyle='--', linewidth=2)
        
        plt.title('Visual Odometry on KITTI Sequence 00', fontsize=15)
        plt.xlabel('X (meters)', fontsize=12)
        plt.ylabel('Z (meters)', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True)
        
        # Save image
        plt.savefig(save_path)
        plt.close()
        print(f"✅ Trajectory plot successfully saved to: {save_path}")

def parse_gt_pose(gt_frame):
    """
    Utility: Converts a row of 12 numbers into a 4x4 transformation matrix
    """
    T = np.eye(4)
    T[:3, :4] = gt_frame.reshape(3, 4)
    return T