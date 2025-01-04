import cv2
import numpy as np
import sys
from concurrent.futures import ThreadPoolExecutor

class Image_Stitching():
    def __init__(self, direction='auto'):
        self.ratio = 0.85
        self.min_match = 10
        self.sift = cv2.SIFT_create()
        self.smoothing_window_size = 800
        self.matcher = cv2.BFMatcher()
        self.direction = direction

    def registration(self, img1, img2):
        # Parallel feature detection using threads
        with ThreadPoolExecutor(max_workers=2) as executor:
            future1 = executor.submit(self.sift.detectAndCompute, img1, None)
            future2 = executor.submit(self.sift.detectAndCompute, img2, None)
            kp1, des1 = future1.result()
            kp2, des2 = future2.result()

        print("Feature detection completed, starting matching...")

        raw_matches = self.matcher.knnMatch(des1, des2, k=2)
        good_points = []
        good_matches = []
        for m1, m2 in raw_matches:
            if m1.distance < self.ratio * m2.distance:
                good_points.append((m1.trainIdx, m1.queryIdx))
                good_matches.append([m1])

        img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_matches, None, flags=2)
        cv2.imwrite('matching.jpg', img3)

        print(f"Found {len(good_points)} matching points")

        if len(good_points) > self.min_match:
            image1_kp = np.float32([kp1[i].pt for (_, i) in good_points])
            image2_kp = np.float32([kp2[i].pt for (i, _) in good_points])
            H, status = cv2.findHomography(image2_kp, image1_kp, cv2.RANSAC, 5.0)
            return H
        else:
            raise ValueError("Not enough matching points to compute homography matrix")

    def create_mask(self, img1, img2, direction):
        height_img1, width_img1 = img1.shape[:2]
        height_img2, width_img2 = img2.shape[:2]
        
        if direction == 'horizontal':
            height_panorama = max(height_img1, height_img2)
            width_panorama = width_img1 + width_img2
        else:  # vertical
            height_panorama = height_img1 + height_img2
            width_panorama = max(width_img1, width_img2)
        
        offset = int(self.smoothing_window_size / 2)
        
        mask = np.zeros((height_panorama, width_panorama))
        
        if direction == 'horizontal':
            barrier = width_img1 - int(self.smoothing_window_size / 2)
            mask[:, barrier - offset:barrier + offset] = np.tile(np.linspace(1, 0, 2 * offset).T, (height_panorama, 1))
            mask[:, :barrier - offset] = 1
        else:  # vertical
            barrier = height_img1 - int(self.smoothing_window_size / 2)
            mask[barrier - offset:barrier + offset, :] = np.tile(np.linspace(1, 0, 2 * offset).T, (width_panorama, 1)).T
            mask[:barrier - offset, :] = 1
        
        return cv2.merge([mask, mask, mask])

    def blending(self, img1, img2):
        print("Starting image registration...")
        H = self.registration(img1, img2)
        print("Registration complete, starting image blending...")
        
        height_img1, width_img1 = img1.shape[:2]
        height_img2, width_img2 = img2.shape[:2]
        
        # 判断拼接方向
        if self.direction == 'auto':
            if width_img1 / height_img1 > width_img2 / height_img2:
                self.direction = 'horizontal'
            else:
                self.direction = 'vertical'
        
        if self.direction == 'horizontal':
            height_panorama = max(height_img1, height_img2)
            width_panorama = width_img1 + width_img2
        else:  # vertical
            height_panorama = height_img1 + height_img2
            width_panorama = max(width_img1, width_img2)
        
        print(f"Stitching direction: {self.direction}")
        
        panorama1 = self._create_panorama(img1, self.create_mask(img1, img2, self.direction), height_panorama, width_panorama)
        panorama2 = self._create_panorama(img2, 1 - self.create_mask(img1, img2, self.direction), height_panorama, width_panorama, H)
        
        result = cv2.addWeighted(panorama1, 1, panorama2, 1, 0)

        rows, cols = np.where(result[:, :, 0] != 0)
        min_row, max_row = min(rows), max(rows) + 1
        min_col, max_col = min(cols), max(cols) + 1
        final_result = result[min_row:max_row, min_col:max_col, :]
        
        print("Image blending complete")
        return final_result

    def _create_panorama(self, img, mask, height, width, H=None):
        panorama = np.zeros((height, width, 3))
        if H is None:
            panorama[0:img.shape[0], 0:img.shape[1], :] = img
        else:
            panorama = cv2.warpPerspective(img, H, (width, height))
        return panorama * mask

def main(argv1, argv2, direction='auto'):
    print("Reading images...")
    img1 = cv2.imread(argv1)
    img2 = cv2.imread(argv2)
    print("Images loaded, starting stitching...")
    final = Image_Stitching(direction).blending(img1, img2)
    print("Stitching complete, saving result...")
    cv2.imwrite('panorama.jpg', final)
    print("All done! Result saved as panorama.jpg")

if __name__ == '__main__':
    try:
        if len(sys.argv) == 3:
            main(sys.argv[1], sys.argv[2])
        elif len(sys.argv) == 4:
            direction = sys.argv[3].lower()
            if direction not in ['horizontal', 'vertical', 'auto']:
                raise ValueError("无效的拼接方向。请使用 'horizontal', 'vertical' 或 'auto'。")
            main(sys.argv[1], sys.argv[2], direction)
        else:
            raise IndexError("参数数量不正确")
    except IndexError:
        print("Please provide two source images and optional stitching direction:")
        print("Example: python Image_Stitching.py '/path/to/image1.jpg' '/path/to/image2.jpg' [horizontal|vertical|auto]")
    except Exception as e:
        print(f"Error occurred: {str(e)}")
