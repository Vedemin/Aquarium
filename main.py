import numpy as np
import os
import cv2
import math

overhead = 10
max_depth = 300
scale = 1

seabed = cv2.imread("wall_test.png")
seabed = 255 - seabed
seabed = cv2.resize(seabed, (seabed.shape[0] * scale, seabed.shape[1] * scale), interpolation=cv2.INTER_AREA)
seabed = cv2.cvtColor(seabed, cv2.COLOR_BGR2GRAY).astype('float64')
seabed *= (max_depth/seabed.max())
seabed = seabed.astype('int64') + overhead

print(seabed.shape)
print(seabed)
print(seabed[50][50])
# We now have a numpy array that carries data of the terrain, scaled to our depth and with an overhead

# [23, -5, 17]

def ray_cast_precise(position, vector):
    magnitute = math.sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)
    currentVector = position.copy()
    bitVector = [x / (magnitute * 2) for x in vector]
    print(bitVector, magnitute)
    for i in range(int(magnitute * 2)):
        currentVector[0] += bitVector[0]
        currentVector[1] += bitVector[1]
        currentVector[2] += bitVector[2]
        x = round(currentVector[0])
        y = round(currentVector[1])
        z = round(currentVector[2])
        print(seabed[z][x], y)
        if seabed[z][x] <= y:
            return False
    return True

def ray_cast(position, direction, distance):
    # Position: Vector3
    # Direction: Vector3
    # Distance: int
    # print(direction)
    free = 0
    for i in range(distance):
        position[0] += direction[0]
        position[1] -= direction[1]
        position[2] += direction[2]
        # print(free, seabed[position[0]][position[2]], position)
        if 0 <= position[0] < seabed.shape[0] and 0 <= position[2] < seabed.shape[1]:
            # print(free, seabed[position[0]][position[2]])
            if seabed[position[0]][position[2]] > position[1]:
                free += 1
            else:
                # print("Wall found", direction, free, position[1], seabed[position[0]][position[2]])
                return free
        else:
            # print("Out of bounds", direction, free)
            return free
    # print("Wall not found - upwards shot", direction, free)
    return free


def get_directions(position, distance):
    # Position is a tuple of (x, y, z)
    # Distance is a float
    res = np.zeros((3, 3, 3))

    for x in range(-1, 2):
        for y in range(-1, 2):
            for z in range(-1, 2):
                ray_pos = position.copy()
                res[x][y][z] = ray_cast(ray_pos, [x, y, z], distance)

    return res


pos1 = [80, 100, 80]
pos2 = [120, 100, 80]
vec = [pos2[0] - pos1[0], pos2[1] - pos1[1], pos2[2] - pos1[2]]

# print(get_directions([100, 150, 100], 700))

print(vec)
print(ray_cast_precise(position=pos1, vector=vec))