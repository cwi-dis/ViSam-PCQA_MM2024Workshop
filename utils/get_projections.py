import numpy as np
import time
import open3d as o3d
import os
from PIL import Image
import cv2
import argparse
import matplotlib.pyplot as plt


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def background_crop(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print(gray_img.shape)
    col = np.mean(gray_img, axis=0)
    row = np.mean(gray_img, axis=1)
    for i in range(len(col)):
        if col[i] != 255:
            col_a = i
            break
    for i in range(len(col)):
        if col[-i] != 255:
            col_b = len(col) - i
            break
    for i in range(len(row)):
        if row[i] != 255:
            row_a = i
            break
    for i in range(len(row)):
        if row[-i] != 255:
            row_b = len(row) - i
            break
    # cv2.imshow('image_Projected',img)
    # plt.imshow(img)
    # plt.show()
    img = img[row_a:row_b, col_a:col_b, :]
    # plt.imshow(img)
    # plt.show()
    # cv2.imshow('image_BackgroudRemoval',img)
    # cv2.waitKey(0)
    return img, row_a, row_b, col_a, col_b


def depth_map_crop(depth_img, row_a, row_b, col_a, col_b):
    depth_img_cropped = depth_img[row_a:row_b, col_a:col_b]
    return depth_img_cropped


def generate_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path


# Camera Rotation
def camera_rotation(type, path, img_path, dep_path):
    print(path)
    if type == "ply":
        obj = o3d.io.read_point_cloud(path)
        normalized_points = pc_normalize(np.array(obj.points))
        obj.points = o3d.utility.Vector3dVector(normalized_points)
        obj.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        print(obj)
        print(np.asarray(obj.points))
        o3d.visualization.draw_geometries([obj])
        obj_with_normals = o3d.geometry.PointCloud()
        obj_with_normals.points = obj.points
        obj_with_normals.colors = obj.normals
    elif type == "mesh":
        obj = o3d.io.read_triangle_mesh(path, True)
        obj.compute_vertex_normals()

    if not os.path.exists(img_path + "/"):
        os.mkdir(img_path + "/")

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=1080, height=1920)
    vis.add_geometry(obj)
    ctrl = vis.get_view_control()

    vis_normal = o3d.visualization.Visualizer()
    vis_normal.create_window(visible=False, width=1080, height=1920)
    vis_normal.add_geometry(obj_with_normals)
    ctrl_normal = vis_normal.get_view_control()

    interval = 5.82  # interval for 1 degree
    start = time.time()
    # begin rotation rotate the camera on the pathway of (x^2 + y^2 = r^2, z = 0) and (y^2 + z^2 = r^2, x = 0)
    rotate_para = [
        [0, 0],
        [90 * interval, 0],
        [90 * interval, 0],
        [90 * interval, 0],
        [90 * interval, 90 * interval],
        [180 * interval, 0 * interval],
    ]
    for i in range(6):
        ctrl.rotate(rotate_para[i][0], rotate_para[i][1])
        vis.poll_events()
        vis.update_renderer()
        img = vis.capture_screen_float_buffer(True)
        img = Image.fromarray((np.asarray(img) * 255).astype(np.uint8))
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        # Capture the depth image
        img_depth = vis.capture_depth_float_buffer(True)
        # print(np.min(img_depth))
        # print(np.max(img_depth))
        # print(np.asarray(img_depth))
        # crop the main object out of the white background
        img, row_a, row_b, col_a, col_b = background_crop(img)
        img_depth = depth_map_crop(np.asarray(img_depth), row_a, row_b, col_a, col_b)
        # normalize the depth image
        img_depth = (img_depth - np.min(img_depth)) / (
            np.max(img_depth) - np.min(img_depth)
        )
        # print(np.min(img_depth))
        # print(np.max(img_depth))
        # cv2.imwrite(os.path.join(img_path,str(i)+'.png'),img)
        # plt.imsave(os.path.join(dep_path,str(i)+'.png'),np.asarray(img_depth))

        # capture the normal image
        ctrl_normal.rotate(rotate_para[i][0], rotate_para[i][1])
        vis_normal.poll_events()
        vis_normal.update_renderer()
        img_normal = vis_normal.capture_screen_float_buffer(True)
        print(np.min(img_normal))
        print(np.max(img_normal))
        print(np.asarray(img_normal))
        img_normal = Image.fromarray((np.asarray(img_normal) * 255).astype(np.uint8))
        plt.imshow(img_normal)
        plt.show()
        img_normal = cv2.cvtColor(np.asarray(img_normal), cv2.COLOR_RGB2BGR)

    end = time.time()
    print("time consuming: ", end - start)
    vis.destroy_window()
    del ctrl
    del vis


def projection(type, path, img_path, dep_path):
    # find all the objects
    objs = os.walk(path)
    cnt = 0
    for path, dir_list, file_list in objs:
        cnt = cnt + 1
        for obj in file_list:
            # for textured mesh
            if type == "mesh":
                # for tmq source
                if obj.endswith(".obj") in path:
                    one_object_path = os.path.join(path, obj)
                    camera_rotation(
                        type, one_object_path, generate_dir(os.path.join(img_path, obj))
                    )
                else:
                    continue
            # for colored point clouds
            elif type == "ply":
                one_object_path = os.path.join(path, obj)
                camera_rotation(
                    type,
                    one_object_path,
                    generate_dir(os.path.join(img_path, obj)),
                    generate_dir(os.path.join(dep_path, obj)),
                )


if __name__ == "__main__":
    # capture the projections of the 3d model
    parser = argparse.ArgumentParser()

    parser.add_argument("--type", type=str, default="ply")  # the format of the 3D model
    parser.add_argument(
        "--path", type=str, default=""
    )  # path to the file that contain .ply models C:\Xuemei\XuemeiZhou\datasets_formatted\SJTU\stimuli
    parser.add_argument(
        "--img_path", type=str, default=""
    )  # path to the generated 2D input
    parser.add_argument(
        "--dep_path", type=str, default=""
    )  # path to the generated 2D depth image input
    parser.add_argument(
        "--nor_path", type=str, default=""
    )  # path to the generated 2D normal image input

    config = parser.parse_args()
    projection(config.type, config.path, config.img_path, config.dep_path)
