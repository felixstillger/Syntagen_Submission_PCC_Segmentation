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
import argparse

warnings.filterwarnings("ignore")

from utils.dataset_config import pascal_palette, class_index_dict


def parse_args():
    parser = argparse.ArgumentParser(description="Clustering and Classification")
    parser.add_argument("source_folder", type=str, help="Path to the source folder")
    parser.add_argument("--start_idx", "-s", help="Start", type=int, default=0)
    parser.add_argument("--end_idx", "-e", help="Start", type=int, default=10000)
    args = parser.parse_args()

    return args


def main(args):

    # 3 different confidence levels
    CONFIDENCES = [0.3, 0.5, 0.8]

    # 3 different numbers of clusters to create
    CLUSTER_NUMS = [4, 7, 10]

    data_path = args.source_folder
    target_dir = os.path.join(data_path, "blocky_mask")

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # load the data
    activation_path = os.path.join(data_path, "activation_data")
    cluster_path = os.path.join(data_path, "cluster_data")

    # Iterate over all files
    for example in range(int(args.start_idx), int(args.end_idx)):

        try:
            # load all first files of all lists
            with open(
                f"{activation_path}/attentionmap_{example}_descriptive.pickle", "rb"
            ) as f:
                activation = pickle.load(f)
                class_names = pickle.load(f)
        except:
            print(f"index {example} does not exist")

        # remove the last token "<EoT>" from names and activation
        activation = np.array(activation)[:-1, ...]
        class_names.pop()

        # replace the first token with "background"
        class_names.pop(0)
        class_names.insert(0, "background")

        activation_64x64 = []
        for layer in activation:
            # OVAM outputs are 512x512, reduce them to 64x64 to match the cluster data
            acc_layer = skimage.measure.block_reduce(layer, (8, 8), np.sum)
            activation_64x64.append(acc_layer)

        activation_64x64 = np.asarray(activation_64x64)

        class_wise_activations = np.zeros((21, 64, 64))

        for layer_idx, class_name in enumerate(class_names):
            # get the index of the class
            class_index = class_index_dict[class_name]

            class_wise_activations[class_index] = np.maximum(
                activation_64x64[layer_idx], class_wise_activations[class_index]
            )

        # Load the self attention PCA data
        with open(
            f"{cluster_path}/cluster_{example}_all_attention_pca.pickle", "rb"
        ) as f:
            cluster = pickle.load(f)

        # pixel position map
        x = np.arange(0, 64)
        y = np.arange(0, 64)
        X, Y = np.meshgrid(x, y)

        # rescale cluster data to 0-1 (dimension-wise)
        # cluster if of shape (4096, 432) (position, dimension)
        cluster_norm_dim = (cluster - cluster.min(axis=0)) / (
            cluster.max(axis=0) - cluster.min(axis=0)
        )

        # rescale the X and Y data to 0-1
        X_norm = (X - X.min()) / (X.max() - X.min())
        Y_norm = (Y - Y.min()) / (Y.max() - Y.min())

        # add the position data to the cluster data
        cluster_norm_dim_pos = np.concatenate(
            (cluster_norm_dim, X_norm.reshape(-1, 1), Y_norm.reshape(-1, 1)), axis=-1
        )

        cluster_seg_maps = []

        # For each cluster number create the clusters from the PCA + position data
        for cluster_idx, cluster_num in enumerate(CLUSTER_NUMS):

            # KMeans clustering
            kmeans = KMeans(n_clusters=cluster_num, random_state=0).fit(
                cluster_norm_dim_pos
            )
            cluster_labels = kmeans.labels_
            cluster_labels += 1  # 0 is background
            clustered_img = cluster_labels.reshape(64, 64)
            clustered_img_new = np.zeros_like(clustered_img)

            # Split the clusters into smaller clusters if they are not connected
            for _, cluster_id in enumerate(np.unique(clustered_img)):

                cur_cluster_binary = np.copy(clustered_img)
                cur_cluster_binary[clustered_img != cluster_id] = 0

                seperated_cluster_img, _ = scipy.ndimage.label(
                    cur_cluster_binary, structure=np.ones((3, 3))
                )

                seperated_cluster_img += np.max(clustered_img_new)

                clustered_img_new[clustered_img == cluster_id] = seperated_cluster_img[
                    clustered_img == cluster_id
                ]

            # reindex clustered_img_new (not efficient, sue me)
            second_clustered_img_new = np.zeros_like(clustered_img_new)
            add_to = 0
            for idx, cluster_id in enumerate(np.unique(clustered_img_new)):
                if cluster_id == 0:
                    continue
                if idx == 0:
                    add_to = 1
                second_clustered_img_new[clustered_img_new == cluster_id] = idx + add_to

            # Get the OVAM activations for each class

            # Reshape the class_wise_activations to be (64, 64, 21)
            # Then rescale the activations to be between 0 and 1
            if class_wise_activations.shape[0] != 64:
                class_wise_activations = np.moveaxis(class_wise_activations, 0, -1)

                class_wise_activations = class_wise_activations - np.min(
                    class_wise_activations, axis=(0, 1)
                )
                class_wise_activations = class_wise_activations / np.max(
                    class_wise_activations, axis=(0, 1)
                )
            class_wise_activations[np.isnan(class_wise_activations)] = 0

            all_conf_semseg = []

            # For each confidence level create a semantic segmentation
            for map_idx, map_size in enumerate(CONFIDENCES):

                all_iou_img = np.zeros((21, 64, 64))

                for class_index in range(21):
                    current_class_activation = class_wise_activations[..., class_index]

                    # skip if empty
                    if np.sum(current_class_activation) == 0:
                        continue

                    if class_index == 0:
                        # SOT is overly strong, limit it a bit
                        map = current_class_activation > 1.2 * map_size

                    else:
                        map = current_class_activation > map_size

                    # get all cluster IDs that overlap with class pixels over the threshhold
                    map_clusters = np.unique(second_clustered_img_new[map])

                    iou_img = np.zeros_like(second_clustered_img_new).astype(np.float32)

                    # calc the IoU of each cluster with the class pixels
                    for cluster_id in map_clusters:
                        iou = np.sum(
                            np.logical_and(map, second_clustered_img_new == cluster_id)
                        ) / (
                            np.sum(map)
                            + np.sum(second_clustered_img_new == cluster_id)
                            - np.sum(
                                np.logical_and(
                                    map, second_clustered_img_new == cluster_id
                                )
                            )
                        )

                        # Assign the IoU with the current class to all cluster pixels
                        iou_img[second_clustered_img_new == cluster_id] = iou

                    # add the class IoU image it's class position in the all_iou_img array
                    all_iou_img[class_index] = iou_img

                # Assign the class with the highest IoU to each pixel
                semseg = np.argmax(all_iou_img, axis=0)

                # append the semseg to the list of semsegs (one for each confidence level)
                all_conf_semseg.append(semseg)

            # append the semseg to the list of semsegs (one for each cluster number)
            cluster_seg_maps.append(all_conf_semseg)

        # Flatten the list of lists
        flat_list = [array for sublist in cluster_seg_maps for array in sublist]

        # Stack the arrays into a single 3D array (64, 64, N_CONFIDENCES * N_CLUSTERS)
        combined_array = np.stack(flat_list, axis=2)

        # Get the most frequent class for each pixel in the combined array
        mode_result = mode(combined_array, axis=2)
        most_frequent_values = mode_result.mode
        mode_count = mode_result.count

        # if less than 5/9 semseg images agree on the class, set the pixel to uncertain
        class_share = mode_count / combined_array.shape[-1]
        most_frequent_values[class_share < 0.56] = 255

        # upscale to 512x512
        blocky_mask = cv2.resize(
            most_frequent_values.astype(np.uint8),
            (512, 512),
            0,
            0,
            interpolation=cv2.INTER_NEAREST,
        )

        # save the current_image and the semseg_full mask
        mask = Image.fromarray(np.array(blocky_mask).astype(np.uint8)).convert("P")
        mask.putpalette(pascal_palette)
        mask.save(f"{target_dir}/img_{example}.png")


if __name__ == "__main__":

    args = parse_args()

    main(args)
