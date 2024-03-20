import numpy as np
import math
from PIL import Image
img = Image.open('peppers.png')
image_3d = np.asarray(img)
print(image_3d)
# height, width, channels = image_array.shape
height, width, channels = image_3d.shape
image_2d = np.zeros((height * width, channels + 2))

for i in range(height):
    for j in range(width):
        idx = i * width + j
        image_2d[idx, 0:3] = image_3d[i, j]
        image_2d[idx, 3:5] = [i, j]
print(image_2d)

pixels = []
for i in  range(len(image_2d)):
    pixels.append(list(image_2d[i]))
print(pixels)
# for y in range(height):
#     for x in range(width):
#         r, g, b = image_array[y][x]
#         pixels.append((r, g, b, x, y))

pixel = [list(t) for t in pixels]
copy_pixel = pixel.copy()
# new_pixels = []
# #print(pixel)
visited = []
bandwidth = [100,50]
class meanshift:
    def isconverged(pixel):
        for i in pixel:
            if i not in visited:
                return 0
        return 1
    def find_rgb(x,y):
        for i in pixel:
            if(i[3]==x and i[4]==y):
                return i[:3]
    def random_pixel(pixel):
        i = np.random.randint(len(pixels))
        if(pixels[i] in visited):
            meanshift.random_pixel(pixel)
        return pixel[i][3:]
    def manhattan_distance(r1,g1,b1,x1,y1,r2,g2,b2,x2,y2):
        color = int(abs(int(r1)-int(r2)) + abs(int(g1)-int(g2)) + abs(int(b1)-int(b2)))
        distance = abs(x1-x2) + abs(y1-y2)
        return [float(distance),(color)]
    def clustering(Pixels, Centroid):
        Centroid = meanshift.random_pixel(Pixels)
        print(Centroid)
        rgb_centre = meanshift.find_rgb(Centroid[0],Centroid[1])
        print(rgb_centre)
        visited.append(rgb_centre+Centroid)
        count = 0
        while True:
            cluster = []
            for i in Pixels:
                if i not in visited:
                    distances = meanshift.manhattan_distance(i[0],i[1],i[2],i[3],i[4],rgb_centre[0],rgb_centre[1],rgb_centre[2],Centroid[0],Centroid[1])
                    if(distances[0]<bandwidth[0] and distances[1]<bandwidth[1]):# and distances[1]<bandwidth[1]):
                        cluster.append(i)
                        
            visited.extend(cluster)
            if(len(cluster)!=0):
                Centroid_new=[0,0]
                for i in cluster:
                    Centroid_new[0]+=i[3]
                    Centroid_new[1]+=i[4]
                Centroid_new[0]//=len(cluster)
                Centroid_new[1]//=len(cluster)
                rgb = meanshift.find_rgb(Centroid_new[0],Centroid_new[1])
                for i in copy_pixel:
                    if i[3]==Centroid[0] and i[4]==Centroid[1]:
                        i[0] = rgb[0]
                        i[1] = rgb[1]
                        i[2] = rgb[2]
                if math.sqrt((Centroid_new[0] - Centroid[0])**2 + (Centroid_new[1] - Centroid[1])**2) < 50 or count == 100:
                    break
                else:
                    Centroid = Centroid_new
            count += 1

            
meanshift.clustering(pixel,[0,0])
print(pixel)
# print(len(copy_pixel))
# Pixels = []
# for i in pixels:
#     Pixels.append(i[:3])
p = []
for i in copy_pixel:
    x = []
    for j in i:
        x.append(int(j))
    p.append(x)
data_2d = np.array(p).reshape((960, 1024))
print(data_2d.size)
print(image_3d.size)
data_3d = data_2d[:512*768].reshape((512, 768, 3))
# segmented_image = Image.fromarray(data_3d)
# segmented_image.save('finalimage.png')