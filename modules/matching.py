import cv2

class FeatureMatcher:
    def __init__(self):
        # Note: crossCheck must be False when using KNN matching
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def match_features(self, desc1, desc2):
        """
        Perform high-quality feature matching using KNN and Lowe's Ratio Test
        """
        # Find the 2 nearest neighbors for each feature point (k=2)
        matches = self.bf.knnMatch(desc1, desc2, k=2)
        
        good_matches = []
        # Lowe's Ratio Test
        for m, n in matches:
            # If the distance of the closest match is less than 75% of the second closest 
            # (0.75 is a rule of thumb), the match is considered distinctive.
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        
        return good_matches

    def draw_matches(self, img1, kp1, img2, kp2, matches):
        match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, 
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        return match_img

    def save_matching_gif(self, matching_imgs, save_path):
        import imageio
        import cv2
        
        if not matching_imgs:
            print("❌ No matching images to save.")
            return

        print(f"🎬 Creating clean GIF with imageio...")
        
        rgb_frames = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in matching_imgs]
        
        imageio.mimsave(save_path, rgb_frames, duration=0.05, loop=0)
        
        print(f"✅ Clean Matching GIF saved to: {save_path}")