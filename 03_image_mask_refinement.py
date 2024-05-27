import os
import glob
import sys
import pickle
import time
import argparse

import cv2
import numpy as np
import scipy
from scipy.stats import mode

from sklearn.cluster import KMeans
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import skimage.measure

import warnings

warnings.filterwarnings("ignore")

from utils.dataset_config import pascal_palette,class_index_dict ,classes
from utils.dataset_config import classes as classes_pascal


def parse_args():
    parser = argparse.ArgumentParser(description="Clustering and Classification")
    parser.add_argument("source_folder", type=str, help="Path to the source folder")
    parser.add_argument("--start_idx", "-s", help="Start", type=int, default=0)
    parser.add_argument("--end_idx", "-e", help="Start", type=int, default=10000)
    args = parser.parse_args()

    return args


def main(args):

    data_path = args.source_folder

    # check if the source folder exists
    if not os.path.exists(os.path.join(data_path, "blocky_mask")):
        print("Source folder (blocky mask) does not exist")
        sys.exit()

    target_dir = os.path.join(data_path, "refined_mask")

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # load the data
    image_path = os.path.join(data_path, "image")
    mask_path = os.path.join(data_path, "blocky_mask")

    # Iterate over all files
    for image_index in range(int(args.start_idx), int(args.end_idx)):

        # load the image
        image = np.array(Image.open(f"{image_path}/img_{image_index}.jpg"))

        # load the mask (palette mode)
        orig_mask = Image.open(f"{mask_path}/img_{image_index}.png")
        orig_mask = np.array(
            orig_mask.convert("P", palette=Image.ADAPTIVE, colors=len(classes_pascal))
        )

        # get the x and y coordinates of each pixel
        x, y = np.meshgrid(range(image.shape[1]), range(image.shape[0]))

        # rescale the x and y coordinates from 0-512 to 0-255
        x = x / 512.0 * 255.0
        y = y / 512.0 * 255.0

        # image to point cloud
        img_pc = image.reshape(-1, 3)

        # add the x and y coordinates to the image (512x512 image -> 512*512 points of dimension 5 (r, g, b, x, y))
        img_pc = np.hstack([img_pc, x.reshape(-1, 1), y.reshape(-1, 1)])

        # knn clustering and visualization
        kmeans = KMeans(n_clusters=20, random_state=0).fit(img_pc)
        cluster_labels = kmeans.labels_
        cluster_labels += 1  # 0 is background
        clustered_img = cluster_labels.reshape(512, 512)
        clustered_img_new = np.zeros_like(clustered_img)

        for _, cluster_id in enumerate(np.unique(clustered_img)):

            cur_cluster_binary = np.copy(clustered_img)
            cur_cluster_binary[clustered_img != cluster_id] = 0

            # seperate non-connected clusters
            seperated_cluster_img, _ = scipy.ndimage.label(
                cur_cluster_binary, structure=np.ones((3, 3))
            )

            seperated_cluster_img += np.max(clustered_img_new)

            clustered_img_new[clustered_img == cluster_id] = seperated_cluster_img[
                clustered_img == cluster_id
            ]

        # remove all clusters that are smaller than x pixels
        cluster_ids, id_count = np.unique(clustered_img_new, return_counts=True)
        for i, cluster_id in enumerate(cluster_ids):
            if id_count[i] < 10:
                clustered_img_new[clustered_img_new == cluster_id] = 0

        # reindex the cluster ids for both images
        cluster_ids = np.unique(clustered_img_new)
        for i, cluster_id in enumerate(cluster_ids):
            clustered_img_new[clustered_img_new == cluster_id] = i

        semseg_assignment_img = np.zeros((512, 512))
        for i, cluster_id in enumerate(np.unique(clustered_img_new)):
            # get the mask of the current cluster
            cluster_mask = clustered_img_new == cluster_id

            # get the semseg of the mask
            classes, classes_count = np.unique(
                np.array(orig_mask)[cluster_mask], return_counts=True
            )

            sort = classes_count.argsort()

            classes = classes[sort[::-1]]
            classes_count = classes_count[sort[::-1]]

            # If only a single class is present in the current image KMeans Cluster, assign it
            if len(classes) == 1:
                semseg_assignment_img[cluster_mask] = classes[0]

            # If there is more than one class, assign the most prominent one if it is assigned to more than 66% of the pixels
            else:
                if classes_count[0] > 2 * classes_count[1]:
                    semseg_assignment_img[cluster_mask] = classes[0]
                else:
                    # If not, assign uncertain label
                    semseg_assignment_img[cluster_mask] = 255

        # fill small "uncertainty" holes in the segmentation masks (mostly due to image KMeans artifacts)
        holes_filled = np.zeros((512, 512))
        for class_val in np.unique(semseg_assignment_img):

            if class_val in [0, 255]:
                holes_filled[semseg_assignment_img == class_val] = class_val
                continue

            cur_class_mask = (semseg_assignment_img == class_val).astype(np.uint8) * 255

            # kernel
            kernel_3 = np.ones((3, 3), np.uint8)

            # remove noise
            cur_class_mask = cv2.erode(cur_class_mask, kernel_3, iterations=1)

            # fill holes
            cur_class_mask = cv2.dilate(cur_class_mask, kernel_3, iterations=5)

            # back to normal size
            cur_class_mask = cv2.erode(cur_class_mask, kernel_3, iterations=4)

            # uncertainty mask
            uncertainty = cv2.dilate(cur_class_mask, kernel_3, iterations=2)

            holes_filled[uncertainty == 255] = 255
            holes_filled[cur_class_mask == 255] = class_val

        holes_filled = Image.fromarray(holes_filled.astype(np.uint8)).convert("P")
        holes_filled.putpalette(pascal_palette)

        # save the current_image and the semseg_full mask
        mask = Image.fromarray(np.array(holes_filled).astype(np.uint8)).convert("P")
        mask.putpalette(pascal_palette)
        mask.save(f"{target_dir}/img_{image_index}.png")


if __name__ == "__main__":

    args = parse_args()

    main(args)
