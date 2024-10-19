import numpy as np
import open3d as o3d
import torch
import random
# Load the point cloud from the .npy file
point_cloud_list = list(np.load("Z:/jpei/MM-PCQA/sjtu_patch_with_normal_2048/longdress_05.npy"))
totoal_num = len(point_cloud_list)
print("totoal_num")
print(totoal_num)
npoint = 2048
selected_patches = np.zeros([6, npoint])  
random_patches = random.sample(point_cloud_list, 6)
num_colors = totoal_num

# Initialize an empty list to store the colors
color_list = []

# Generate random RGB values for each color
for _ in range(num_colors):
    red = random.uniform(0, 1)
    green = random.uniform(0, 1)
    blue = random.uniform(0, 1)
    color_list.append([red, green, blue])

# Print the generated color list
print(color_list)

for i in range(totoal_num):
    selected_patches = np.asarray_chkfinite(point_cloud_list[i])
    # Extract the xyz and normal of each point
    xyz = selected_patches[:, :3]
    print("xyz.shape")
    print(xyz.shape)
    
    normal = selected_patches[:, 3:]
    print("normal.shape")
    print(normal.shape)
    patch_point_clouds = o3d.geometry.PointCloud()
    patch_point_clouds.points = o3d.utility.Vector3dVector(xyz)
    patch_point_clouds.normals = o3d.utility.Vector3dVector(normal)
    patch_point_clouds.paint_uniform_color(color_list[i])

    # Set camera view to zoom out
    # Create a visualizer with editing
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False,width=512, height=512)
    vis.add_geometry(patch_point_clouds)
    # Get the view control
    view_control = vis.get_view_control()
    # Zoom out by changing the view control parameters
    view_control.scale(0.1)  # Adjust the scale factor as needed
    
    # Visualize the point clouds
    # o3d.visualization.draw_geometries([patch_point_clouds])
    #Save screenshot
    o3d.visualization.draw_geometries_with_animation_callback(
    [patch_point_clouds],
    lambda vis: vis.capture_screen_image(f"C:/Xuemei/2023CVPR/imgs/longdress_05_patch_{i}.png"),
    )
    # Close the visualizer
    vis.destroy_window()




