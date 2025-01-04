# Image_stitching

A Python tool for stitching two images together using SIFT features and homography transformation.

## Features
- Automatic detection of stitching direction
- Support for both horizontal and vertical stitching
- Multi-threaded feature detection
- Smooth blending at image boundaries

## Implementation Details

### 1. Feature Detection and Matching
- Uses SIFT (Scale-Invariant Feature Transform) algorithm for feature detection
- Implements parallel processing for feature detection using ThreadPoolExecutor
- Applies ratio test for matching features between images
- Visualizes matching points in 'matching.jpg'

### 2. Image Registration
- Computes homography matrix using RANSAC algorithm
- Requires minimum 10 matching points for reliable transformation
- Transforms the second image to align with the first image's perspective

### 3. Direction Detection
- Automatically determines optimal stitching direction based on image aspect ratios
- Supports manual override for horizontal or vertical stitching
- Calculates appropriate output dimensions based on stitching direction

### 4. Blending
- Creates smooth transitions between images using gradient masks
- Implements weighted blending at the seam
- Automatically crops the final panorama to remove empty regions

## Usage

```bash
python Image_Stitching.py '/path/to/image1.jpg' '/path/to/image2.jpg' [horizontal|vertical|auto]
```

### Parameters
- `image1.jpg`: First input image
- `image2.jpg`: Second input image
- `direction`: Optional stitching direction (default: auto)
  - `auto`: Automatically determine direction
  - `horizontal`: Force horizontal stitching
  - `vertical`: Force vertical stitching

### Output
- `matching.jpg`: Visualization of matched features between images
- `panorama.jpg`: Final stitched panorama

## Limitations
- Requires overlapping regions between images
- Best results with images taken from similar viewpoints
- Performance depends on the quality of feature matches