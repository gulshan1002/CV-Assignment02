# Gulshan Kumar
# S20200010071

import cv2
import numpy as np

# Load the two images
img1 = cv2.imread('img2.png')
img2 = cv2.imread('img4.png')

# Convert the images to grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Detect SIFT keypoints and compute descriptors for both images
sift = cv2.xfeatures2d.SIFT_create()
keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

# Create a brute-force matcher object
bf = cv2.BFMatcher()

# Match the descriptors of the two images using the brute-force matcher
matches = bf.match(descriptors1, descriptors2)

# Sort the matches by distance
matches = sorted(matches, key=lambda x:x.distance)

# Compute the sum of squared differences (SSD) and ratio distance for each match
ssd_list = []
ratio_list = []
for match in matches:
    ssd = np.sum(np.square(descriptors1[match.queryIdx] - descriptors2[match.trainIdx]))
    ssd_list.append(ssd)
    if ssd > 0:
        ratio_list.append(np.sum(np.square(descriptors1[match.queryIdx] - descriptors2[match.trainIdx+1])) / ssd)
    else:
        ratio_list.append(np.inf)

# Get the indices of the best matches based on ratio distance
best_matches_idx = np.argsort(ratio_list)[:10]

# Draw the keypoints and the best matches on the images
img_with_keypoints1 = cv2.drawKeypoints(img1, keypoints1, None, color=(0,255,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img_with_keypoints2 = cv2.drawKeypoints(img2, keypoints2, None, color=(0,255,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img_with_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Show the images
cv2.imshow('S20200010071_FDDM_output3', img_with_matches)
cv2.imshow('S20200010071_FDDM_output2', img_with_keypoints2)
cv2.imshow('S20200010071_FDDM_output1', img_with_keypoints1)

cv2.imwrite('S20200010071_FDDM_output1.png', img_with_keypoints1)
cv2.imwrite('S20200010071_FDDM_output2.png', img_with_keypoints2)
cv2.imwrite('S20200010071_FDDM_output3.png', img_with_matches)
cv2.waitKey(0)
cv2.destroyAllWindows()
