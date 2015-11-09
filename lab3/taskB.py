# coding=utf-8
import cv2
import numpy as np
from sklearn.cluster import KMeans

__author__ = 'Michał Ciołczyk'


def read_image(name):
    img = cv2.imread(name)
    shape = img.shape
    img = img.reshape(-1, img.shape[-1])  # Remove alpha
    return np.array(img, int), shape


def recreate_image(kmeans, cluster):
    img = []
    cluster_centers = kmeans.cluster_centers_
    for x in cluster:
        img.append(cluster_centers[x])
    return np.array(img)


def save_image(array, shape, name):
    img = np.array(array)
    cv2.imwrite(name, img.reshape(shape))


if __name__ == "__main__":
    img, shape = read_image("partB/input.png")
    for k in [2, 4, 8, 16, 32]:
        kmeans = KMeans(k).fit(img)
        closest_cluster = kmeans.predict(img)
        output = recreate_image(kmeans, closest_cluster)
        save_image(output, shape, "partB/output" + str(k) + ".png")
