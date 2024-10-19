import open3d as o3d
# Load PLY file
pcd = o3d.io.read_point_cloud("C:/Xuemei/XuemeiZhou/datasets_formatted/SJTU/stimuli_fused/longdress_05.ply")

# Compute normals
pcd.estimate_normals()
# generate another point cloud object with only xyz and normal
pcd_xyz_normal = o3d.geometry.PointCloud()
pcd_xyz_normal.points = pcd.points
pcd_xyz_normal.normals = pcd.normals
pcd_xyz_normal.paint_uniform_color([0.5, 0.5, 0.5])
# Visualize point cloud with normals
# Create a visualizer object
vis = o3d.visualization.Visualizer()
vis.create_window()

# Add point cloud with normals to the visualizer
vis.add_geometry(pcd_xyz_normal)
# Visualize the point cloud with normals
vis.run()
vis.destroy_window()

# Save screenshot
o3d.visualization.draw_geometries_with_animation_callback(
    [pcd_xyz_normal],
    lambda vis: vis.capture_screen_image("C:/Xuemei/2023CVPR/imgs/longdress_05.png"),
)
