import os
import cv2
import glob

class KITTIDataset:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        # 假设你的图片放在 data/sequences/00/image_0/ 目录下
        self.image_path = os.path.join(data_dir, 'sequences/00/image_0/*.png')
        self.images = sorted(glob.glob(self.image_path))
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 读取灰度图
        image = cv2.imread(self.images[idx], cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not read image at {self.images[idx]}")
        return image

# 测试代码
if __name__ == "__main__":
    # 你可以先在 data 文件夹下放两张测试图跑一下
    print("Dataset module initialized.")