import cv2
from matplotlib import pyplot as plt
import numpy as np

obstacleImage = cv2.imread("map.png")
obstacleImage = cv2.cvtColor(obstacleImage, cv2.COLOR_BGR2GRAY)
grid = cv2.threshold(
    obstacleImage, 127, 255, cv2.THRESH_BINARY)[1]
plt.imshow(grid)
