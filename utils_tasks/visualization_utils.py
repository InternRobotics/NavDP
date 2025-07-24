import numpy as np
import cv2
from collections import deque
from scipy.ndimage import binary_dilation

class VisualizationManager:
    def __init__(self, history_size=5):
        self.history_size = history_size
        self.occupancy_history = deque(maxlen=history_size)  # Will store (grid, min_coords, robot_pose)
        self.resolution = 0.05  # 5cm per pixel
        self.inflation = 5      # inflation radius in pixels
    
    def reset(self):
        self.occupancy_history.clear()
        
    def build_occupancy_grid(self, depth_map, intrinsic, camera_roll=0):
        try:
            """Convert depth image to occupancy grid in BEV"""
            height, width = depth_map.shape
            uu, vv = np.meshgrid(np.arange(width), np.arange(height))
            z = depth_map
            x = (uu - intrinsic[0, 2]) * z / intrinsic[0, 0]
            y = (vv - intrinsic[1, 2]) * z / intrinsic[1, 1]
            
            # Filter valid points
            valid_mask = (z > 0) & np.isfinite(z) & (z < 10)
            points_3d = np.stack((x[valid_mask], y[valid_mask], z[valid_mask]), axis=-1)
            
            # Apply camera roll
            roll = camera_roll * np.pi / 180
            rotation_matrix_x = np.array([[1, 0, 0], 
                                        [0, np.cos(roll), -np.sin(roll)], 
                                        [0, np.sin(roll), np.cos(roll)]])
            point_3d_flat = (rotation_matrix_x @ points_3d.transpose()).transpose()
            
            # Transform to world coordinates
            point_3d_world = np.zeros((point_3d_flat.shape[0], 3))
            point_3d_world[:, 0] = point_3d_flat[:, 2]
            point_3d_world[:, 1] = -point_3d_flat[:, 0]
            point_3d_world[:, 2] = -point_3d_flat[:, 1]
            bins = np.arange(np.min(point_3d_world[:, 2]), np.max(point_3d_world[:, 2]), 0.05)
            try:
                hist, bin_edges = np.histogram(point_3d_world[:, 2], bins=bins)
                max_freq_index = np.argmax(hist)
                point_3d_world[:, 2] -= bin_edges[max_freq_index]
                # print(f"bin_edges[max_freq_index] {bin_edges[max_freq_index]}")
            except:
                point_3d_world[:, 2] -= -0.5
            
            # Filter points within height range
            filtered_points = point_3d_world[(point_3d_world[:, 2] >= 0.2) & (point_3d_world[:, 2] <= 1.5)]
            if filtered_points.shape[0] == 0:
                min_coords = np.array([-5.0,-5.0,-5.0])
                max_coords = np.array([5.0,5.0,5.0])
                grid_size = np.ceil((max_coords - min_coords) / self.resolution + 1).astype(int)
                occupancy_grid = np.zeros(grid_size[:2], dtype=np.int8)
                return occupancy_grid, min_coords
                
            # Create occupancy grid
            min_coords = np.min(filtered_points, axis=0)
            max_coords = np.max(filtered_points, axis=0)
            grid_size = np.ceil((max_coords - min_coords) / self.resolution + 1).astype(int)
            occupancy_grid = np.zeros(grid_size[:2], dtype=np.int8)
            
            grid_coords = ((filtered_points[:, :2] - min_coords[:2]) / self.resolution).astype(int)
            occupancy_grid[grid_coords[:, 0], grid_coords[:, 1]] = 1
            
        except:
            occupancy_grid = np.zeros((100,100),dtype=np.int8)
            min_coords = np.array([0,0])
        
        return occupancy_grid, min_coords
        
    def visualize_trajectory(self, rgb_image, depth_image, intrinsic, trajectory_points, robot_pose, camera_roll=0, all_trajectories_points=None, all_trajectories_values=None):
        # Calculate visualization size based on 10m×10m range
        grid_size = int(10.0 / self.resolution)  # 20m in grid cells
        vis_image = np.zeros((grid_size, grid_size, 3), dtype=np.uint8)

        # Resize visualization to match RGB image height with better interpolation
        vis_resized = cv2.resize(vis_image, (int(rgb_image.shape[0]), int(rgb_image.shape[0])), interpolation=cv2.INTER_CUBIC)
        # Apply slight Gaussian blur to smooth pixelated edges (adjust sigma as needed)
        vis_resized = cv2.GaussianBlur(vis_resized, (3, 3), 0.5)
        
        # Concatenate images
        combined_image = np.concatenate((rgb_image, vis_resized), axis=1)
         
        # Build current occupancy grid
        occupancy_grid, min_coords = self.build_occupancy_grid(depth_image[..., 0], intrinsic, camera_roll)
        if occupancy_grid is None:
            return combined_image
        
        # Add to history with robot pose
        self.occupancy_history.append((occupancy_grid, min_coords, robot_pose))
        
        # Calculate center offset (assuming robot is at center)
        center_offset = grid_size // 2
        
        # Draw historical occupancy grids
        all_hist_world_points_list = []
        current_world_points = np.array([])

        # Process historical frames first
        for i, (hist_grid, hist_min_coords, hist_pose) in enumerate(self.occupancy_history):
            # Get occupied points in the grid's local frame
            grid_coords = np.where(hist_grid > 0)
            points = np.array([
                grid_coords[0] * self.resolution + hist_min_coords[0],
                grid_coords[1] * self.resolution + hist_min_coords[1]
            ]).T
            
            # Transform points from the grid's local frame to world frame
            hist_rotation = np.array([
                [np.cos(hist_pose[2]), -np.sin(hist_pose[2])],
                [np.sin(hist_pose[2]), np.cos(hist_pose[2])]
            ])
            world_points = (hist_rotation @ points.T).T + hist_pose[:2]

            if i == len(self.occupancy_history) - 1:  # Current frame
                current_world_points = world_points
            else:  # Historical frame
                if world_points.size > 0:
                    all_hist_world_points_list.append(world_points)

        # Combine all historical points
        if all_hist_world_points_list:
            all_hist_world_points = np.concatenate(all_hist_world_points_list, axis=0)
        else:
            all_hist_world_points = np.array([])

        # Helper function to transform world points to vis_coords
        def transform_to_vis_coords(world_pts, current_pose, res, offset, size):
            if world_pts.size == 0:
                return np.array([])
                
            # Transform world points to yaw=0 frame centered at current robot position
            dx = world_pts[:, 0] - current_pose[0]
            dy = world_pts[:, 1] - current_pose[1]
            
            current_rotation = np.array([
                [np.cos(0), -np.sin(0)],
                [np.sin(0), np.cos(0)]
            ])
            transformed_points = (current_rotation @ np.vstack([dx, dy])).T
            
            # Convert to grid coordinates relative to center
            center_coords = (transformed_points / res).astype(int)
            
            # Filter points within visualization range
            valid_mask = (np.abs(center_coords[:, 0]) < size//2) & (np.abs(center_coords[:, 1]) < size//2)
            center_coords = center_coords[valid_mask]
            
            # Convert to visualization coordinates (adjust for image coordinate system)
            vis_coords = np.zeros_like(center_coords)
            vis_coords[:, 0] = -center_coords[:, 0] + offset  # Flip x axis
            vis_coords[:, 1] = -center_coords[:, 1] + offset   # Keep y axis
            
            # Final boundary check
            valid_mask = (vis_coords[:, 0] >= 0) & (vis_coords[:, 0] < size) & \
                        (vis_coords[:, 1] >= 0) & (vis_coords[:, 1] < size)
            vis_coords = vis_coords[valid_mask]
            return vis_coords

        # Draw historical points (Gray)
        vis_coords_hist = transform_to_vis_coords(all_hist_world_points, robot_pose, self.resolution, center_offset, grid_size)
        if vis_coords_hist.size > 0:
            vis_image[vis_coords_hist[:, 0], vis_coords_hist[:, 1]] = (128, 128, 128) # Gray

        # Draw current points (Red)
        vis_coords_current = transform_to_vis_coords(current_world_points, robot_pose, self.resolution, center_offset, grid_size)
        if vis_coords_current.size > 0:
            vis_image[vis_coords_current[:, 0], vis_coords_current[:, 1]] = (0, 0, 255) # Red
        
        # Draw trajectory
        if trajectory_points is not None:
            # Transform trajectory points to yaw=0 frame centered at current robot position
            dx = trajectory_points[:, 0] - robot_pose[0]
            dy = trajectory_points[:, 1] - robot_pose[1]
            
            # Rotate points to align with yaw=0 frame
            current_rotation = np.array([
                [np.cos(0), np.sin(0)],
                [np.sin(0), np.cos(0)]
            ])
            transformed_points = (current_rotation @ np.vstack([dx, dy])).T
            
            # Convert to grid coordinates
            grid_points = (transformed_points / self.resolution).astype(int)
            
            # Filter points within range
            valid_mask = (np.abs(grid_points[:, 0]) < grid_size//2) & (np.abs(grid_points[:, 1]) < grid_size//2)
            grid_points = grid_points[valid_mask]
            
            # Convert to visualization coordinates (adjust for image coordinate system)
            vis_points = np.zeros_like(grid_points)
            vis_points[:, 0] = -grid_points[:, 1] + center_offset  # Flip x axis
            vis_points[:, 1] = -grid_points[:, 0] + center_offset   # Keep y axis
            
            # Draw trajectory with anti-aliased lines
            for i in range(len(vis_points) - 1):
                cv2.line(vis_image, tuple(vis_points[i]), tuple(vis_points[i+1]), (0, 255, 0), 2, cv2.LINE_AA)
            # Draw start and end points
            if len(vis_points) > 0:
                # Use larger circles with anti-aliasing for smoother appearance
                cv2.circle(vis_image, tuple(vis_points[0]), 3, (255, 0, 0), -1, cv2.LINE_AA)  # Blue for start
                
                # Draw robot rectangle at start position
                rect_length = 10  # pixels
                rect_width = 5   # pixels
                start_point = (center_offset, center_offset)
                # Get yaw angle from trajectory points
                yaw = -robot_pose[2]
                
                # Calculate rectangle corners
                cos_yaw = np.cos(yaw)
                sin_yaw = np.sin(yaw)
                corners = np.array([
                    [-rect_width/2, -rect_length/2],
                    [rect_width/2, -rect_length/2],
                    [rect_width/2, rect_length/2],
                    [-rect_width/2, rect_length/2]
                ])
                # Rotate corners
                rot_matrix = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
                rotated_corners = (rot_matrix @ corners.T).T + start_point
                
                # Draw rectangle with anti-aliasing
                corners_int = rotated_corners.astype(np.int32)
                cv2.polylines(vis_image, [corners_int], True, (255, 255, 255), 1, cv2.LINE_AA)  # Blue rectangle
                cv2.circle(vis_image, tuple(vis_points[-1]), 3, (0, 0, 255), -1, cv2.LINE_AA)  # Red for end
        
        # Resize visualization to match RGB image height with better interpolation
        vis_resized = cv2.resize(vis_image, (int(rgb_image.shape[0]), int(rgb_image.shape[0])), interpolation=cv2.INTER_CUBIC)
        # Apply slight Gaussian blur to smooth pixelated edges (adjust sigma as needed)
        vis_resized = cv2.GaussianBlur(vis_resized, (3, 3), 0.5)
        # Concatenate images
        combined_image = np.concatenate((rgb_image, vis_resized), axis=1)
        
        # If no all_trajectories_points, return original combined image
        if all_trajectories_points is None or len(all_trajectories_points) == 0:
            return combined_image
        # print(f"all_trajectories_points: {len(all_trajectories_points)}")
        # --- Create additional visualization for all trajectories ---
        # Create a new image for all trajectories visualization
        vis_image_all = np.zeros((grid_size, grid_size, 3), dtype=np.uint8)
        
        # Draw the same occupancy grid
        if vis_coords_hist.size > 0:
            vis_image_all[vis_coords_hist[:, 0], vis_coords_hist[:, 1]] = (128, 128, 128) # Gray
        if vis_coords_current.size > 0:
            vis_image_all[vis_coords_current[:, 0], vis_coords_current[:, 1]] = (0, 0, 255) # Red
            
        # Draw all trajectories with colors based on values
        # Define color mapping function from value to color (blue to red gradient)
        def value_to_color(value, values_min, values_max):
            # Normalize value to [0, 1] based on a fixed range [-2, 0.5]
            fixed_min = -1.2
            fixed_max = 0.2

            # Clamp the value to be within the fixed range
            value = np.clip(value, fixed_min, fixed_max)

            # Normalize value to [0, 1]
            normalized = (value - fixed_min) / (fixed_max - fixed_min)

            # Map to blue (low) -> green (mid) -> red (high)
            if normalized < 0.5:
                # Blue to green
                b = 255 * (1 - 2 * normalized)
                g = 255 * (2 * normalized)
                r = 0
            else:
                # Green to red
                b = 0
                g = 255 * (2 - 2 * normalized)
                r = 255 * (2 * normalized - 1)
            
            return (int(b), int(g), int(r))  # Return BGR color
        
        # Set default colors if no values provided
        if all_trajectories_values is None:
            colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
            trajectory_colors = [colors[idx % len(colors)] for idx in range(len(all_trajectories_points))]
        else:
            # Get min and max values for normalization
            values_min = np.min(all_trajectories_values)
            values_max = np.max(all_trajectories_values)
            # Generate color for each trajectory
            trajectory_colors = [value_to_color(v, values_min, values_max) for v in all_trajectories_values]
        
        for idx, traj in enumerate(all_trajectories_points):
            color = trajectory_colors[idx]
            
            # Transform trajectory points
            dx = traj[:, 0] - robot_pose[0]
            dy = traj[:, 1] - robot_pose[1]
            
            # Rotate points
            transformed_points = (current_rotation @ np.vstack([dx, dy])).T
            
            # Convert to grid coordinates
            grid_points = (transformed_points / self.resolution).astype(int)
            
            # Filter points within range
            valid_mask = (np.abs(grid_points[:, 0]) < grid_size//2) & (np.abs(grid_points[:, 1]) < grid_size//2)
            grid_points = grid_points[valid_mask]
            
            # Convert to visualization coordinates
            vis_points_all = np.zeros_like(grid_points)
            vis_points_all[:, 0] = -grid_points[:, 1] + center_offset
            vis_points_all[:, 1] = -grid_points[:, 0] + center_offset
            
            # Draw trajectory with anti-aliased lines
            for i in range(len(vis_points_all) - 1):
                cv2.line(vis_image_all, tuple(vis_points_all[i]), tuple(vis_points_all[i+1]), color, 1, cv2.LINE_AA)
                
            # Draw start and end points with anti-aliasing
            if len(vis_points_all) > 0:
                cv2.circle(vis_image_all, tuple(vis_points_all[0]), 2, color, -1, cv2.LINE_AA)
        
        # Draw robot position with anti-aliasing
        if len(vis_points) > 0:
            corners_int = rotated_corners.astype(np.int32)
            cv2.polylines(vis_image_all, [corners_int], True, (255, 255, 255), 1, cv2.LINE_AA)  # White robot outline
        
        # Resize all trajectories visualization - align width with rgb_image
        # Get target width (same as rgb_image width)
        target_width = rgb_image.shape[1]
        # Calculate the height to maintain aspect ratio
        target_height = int(vis_image_all.shape[0] * (target_width / vis_image_all.shape[1]))
        # Resize with the calculated dimensions using better interpolation
        vis_resized_all = cv2.resize(vis_image_all, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
        # Apply slight Gaussian blur to smooth pixelated edges
        vis_resized_all = cv2.GaussianBlur(vis_resized_all, (3, 3), 0.5)
        
        # Create black padding to match combined_image width
        if combined_image.shape[1] > target_width:
            # Add black padding to the right
            padding_width = combined_image.shape[1] - target_width
            padding = np.zeros((target_height, padding_width, 3), dtype=np.uint8)
            vis_resized_all = np.concatenate((vis_resized_all, padding), axis=1)
        
        # Stack vertically: combined_image (top) and vis_resized_all (bottom)
        final_combined_image = np.concatenate((combined_image, vis_resized_all), axis=0)
        
        return final_combined_image 