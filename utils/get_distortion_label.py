import open3d as o3d
import numpy as np
import os
import time
import random
import argparse
import pandas as pd


def visualize(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([0.5, 0.5, 0.5])
    o3d.visualization.draw_geometries([pcd])


def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, patch_size):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    npoint = int(N / patch_size) + 1
    if N < npoint:
        idxes = np.hstack(
            (np.tile(np.arange(N), npoint // N), np.random.randint(N, size=npoint % N))
        )
        return point[idxes, :]

    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


def knn_patch(pcd_name, patch_size=2048):
    pcd = o3d.io.read_point_cloud(pcd_name)
    # nomalize pc and set up kdtree
    points = pc_normalize(np.array(pcd.points))
    pcd.points = o3d.utility.Vector3dVector(points)
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    fps_point = farthest_point_sample(points, patch_size)

    point_size = fps_point.shape[0]

    patch_list = []

    for i in range(point_size):
        [_, idx, dis] = kdtree.search_knn_vector_3d(fps_point[i], patch_size)
        # print(pc_normalize(np.asarray(point)[idx[1:], :]))
        patch_list.append(np.asarray(points)[idx[:], :])

    # visualize(all_point(np.array(patch_list)))
    # visualize(point)
    return np.array(patch_list)


def main(config):
    start = time.time()
    objs = os.walk(config.path)
    df = pd.read_csv(config.path)
    stimulis = df["stimulus"]
    file_path = "C:/Xuemei/2023CVPR/MM-PCQA-IG/csvfiles/sjtu_data_info/"
    test_distortion_types = []
    train_distortion_types = []
    for index, name in enumerate(stimulis):
        if "redandblack" in name:
            test_distortion_types.append(df.iloc[index, 2])
    train_distortion_types = test_distortion_types * 8
    # Create a new DataFrame from the array
    df_new_test = pd.DataFrame({"DT": test_distortion_types})
    df_new_train = pd.DataFrame({"DT": train_distortion_types})
    for i_test in range(9):
        test_file_name = "test_" + str(i_test + 1) + ".csv"
        train_file_name = "train_" + str(i_test + 1) + ".csv"
        csv_file_test = file_path + test_file_name
        csv_file_train = file_path + train_file_name
        data_test = pd.read_csv(csv_file_test)
        data_train = pd.read_csv(csv_file_train)
        # Append the new DataFrame to the existing one
        df_combined_test = pd.concat([data_test, df_new_test], axis=1)
        df_combined_train = pd.concat([data_train, df_new_train], axis=1)
        df_combined_test.to_csv(csv_file_test, index=False)
        print("Data appended to the Excel file test.")
        df_combined_train.to_csv(csv_file_train, index=False)
        print("Data appended to the Excel file train.")
    end = time.time()
    print("Consuming seconds /s :" + str(end - start))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path",
        type=str,
        default="C:/Xuemei/XuemeiZhou/datasets_formatted/SJTU/subjective scores/subj_dis.csv",
    )  # path to the file that contain .ply models
    parser.add_argument(
        "--out_path",
        type=str,
        default="C:/Xuemei/2023CVPR/MM-PCQA-IG/csvfiles/sjtu_data_info/",
    )  # path to the output patches
    config = parser.parse_args()
    main(config)
