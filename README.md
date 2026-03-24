# Visual Odometry with Deep Features 🚗📷

![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

A monocular visual odometry pipeline using ORB/Deep features and geometric pose estimation, evaluated on the KITTI Odometry dataset.

## ✨ Key Features

* **Feature Extraction:** Traditional ORB and support for Deep Learning based features (e.g., SuperPoint).
* **Geometric Verification:** Essential Matrix estimation coupled with RANSAC for robust outlier rejection.
* **Trajectory Recovery:** Monocular scale-corrected trajectory reconstruction.
* **Evaluation:** Direct comparison against KITTI Ground Truth trajectories.

## ⚙️ System Pipeline

**Workflow:**

1. **Image Input** -> 2. **Feature Detection & Description** -> 3. **Feature Matching** -> 4. **Epipolar Geometry (Essential Matrix)** -> 5. **Pose Recovery (R & t)** -> 6. **Trajectory Concatenation**.

## 🛠️ Installation

Clone the repository and install the required dependencies:

```bash
git clone [https://github.com/Jingchen-Chen/visual-odometry-project.git]
cd visual-odometry-project
pip install -r requirements.txt
```

## 🚀 Usage

1. Download the [KITTI Odometry dataset (grayscale)](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) and extract it so that sequences are located in `data/dataset/sequences/`.
2. Ensure the `image_dir` path in `main.py` correctly points to your sequence (e.g., Sequence `00`).
3. Run the pipeline:

```Bash
python main.py
```

## 📊 Results

*(Result images will be uploaded here once the evaluation on KITTI sequence 00 is complete)*

### 1. Feature Matching (Frontend)

![Feature Matching](results/matching_demo.png)

### 2. Trajectory Comparison (Estimated vs. Ground Truth)

![Trajectory Result](results/trajectory_00.png)

## 📈 Error Analysis & Drift

As a purely monocular visual odometry system without backend optimization (like Bundle Adjustment or loop closure), scale drift and translation error naturally accumulate over long distances. This is a fundamental challenge in visual SLAM. Future work will focus on integrating deep-learning-based feature matching (e.g., SuperGlue) to improve frontend robustness and mitigate drift.