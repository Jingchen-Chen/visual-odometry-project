import os
import cv2
import glob

class KITTIDataset:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        # Assuming images are located in: data/sequences/00/image_0/
        self.image_path = os.path.join(data_dir, 'sequences/00/image_0/*.png')
        self.images = sorted(glob.glob(self.image_path))
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Read image in grayscale
        image = cv2.imread(self.images[idx], cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not read image at {self.images[idx]}")
        return image

# Test Code
if __name__ == "__main__":
    # You can place two test images in the data folder to run a quick check
    print("Dataset module initialized.")