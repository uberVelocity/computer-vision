import numpy as np
import cv2
from matplotlib import pyplot as plt

# Load color image in grayscale
img = cv2.imread("/home/merkel/Pictures/wallpapers/catbook.jpg", 1)

# Show the image stored in 'img'
cv2.namedWindow('cat and girl', cv2.WINDOW_AUTOSIZE)
cv2.imshow('cat and girl', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Reverse BGR to RGB as you read with OpenCV(BGR) but plot with matplotlib(RGB)
b, g, r = cv2.split(img)
img = cv2.merge([r, g, b])

# Display the image using pyplot
plt.imshow(img, interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()