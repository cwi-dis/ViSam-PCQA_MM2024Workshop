import numpy as np
import open3d as o3d
import torch
import random
# Load the point cloud from the .npy file
point_cloud_list = list(np.load("Z:/jpei/MM-PCQA/sjtu_patch_with_normal_2048/longdress_04.npy"))
totoal_num = len(point_cloud_list)
print("totoal_num")
print(totoal_num)
npoint = 2048
selected_patches = np.zeros([6, npoint])  
for i in range(totoal_num):
    selected_patches = np.asarray_chkfinite(point_cloud_list[i])
    # Extract the xyz and normal of each point
    xyz = selected_patches[:, :3]
    print("xyz.shape")
    print(xyz.shape)
    # print(xyz)
    normal = selected_patches[:, 3:]
    print("normal.shape")
    print(normal.shape)
    patch_point_clouds = o3d.geometry.PointCloud()
    patch_point_clouds.points = o3d.utility.Vector3dVector(xyz)
    patch_point_clouds.normals = o3d.utility.Vector3dVector(normal)
    patch_point_clouds.paint_uniform_color([0.5, 0.5, 0.5])

   
    # Create a visualizer with editing
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(patch_point_clouds)
    # Set camera view to zoom out (if needed)
    view_control = vis.get_view_control()
    view_control.rotate(45.0, 0.0)  # Rotate the view for a better angle
    view_control.scale(0.6)  # Adjust the scale factor as needed
    # Set lighting conditions (optional)
    render_option = vis.get_render_option()
    render_option.light_on = True  # Enable lighting

    #Save screenshot
    vis.capture_screen_image(f"C:/Xuemei/2023CVPR/imgs/longdress_05_patch_{i}.png")
    # Close the visualizer
    vis.destroy_window()

   




