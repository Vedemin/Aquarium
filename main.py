import numpy as np
import os
import cv2

overhead = 10
max_depth = 300

seabed_heightmap = cv2.imread("seabed_heightmap.png")
seabed_f64 = cv2.cvtColor(seabed_heightmap, cv2.COLOR_BGR2GRAY).astype('float64')
seabed_f64 *= (max_depth/seabed_f64.max())
seabed = seabed_f64.astype('int64') + overhead

# We now have a numpy array that carries data of the terrain, scaled to our depth and with an overhead

def RayCast(position, direction, distance):
    # Position: Vector3
    # Direction: Vector3
    # Distance: float


def GetDirections(position, distance):
    # Position is a tuple of (x, y, z)
    # Distance is a float
    results = [] # The following applies