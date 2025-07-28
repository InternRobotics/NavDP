import requests
import io
import json
import argparse
import cv2
from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="A script to run a car control simulation")
parser.add_argument(
    "--scene_dir", type=str, default="./asset_scenes/cluttered_hard")
parser.add_argument(
    "--scene_index", type=int, default=0)
parser.add_argument(
    "--scene_scale", type=float, default=1.0)
parser.add_argument(
    "--stop_threshold", type=float, default=-3.0)
parser.add_argument(
    "--num_envs", type=int, default=1)
parser.add_argument(
    "--num_episodes", type=int, default=100)
parser.add_argument(
    "--speed", type=float, default=0.5)
parser.add_argument(
    "--port", type=int, default=8888)
args_cli = parser.parse_args()
app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app

import omni
import cv2
import carb
import numpy as np
import imageio
import os
import csv
import torch
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from pxr import Usd, Sdf
from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors.camera.utils import create_pointcloud_from_rgbd
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper
from wheeled_robots.controllers.differential_controller import DifferentialController
import torchvision.transforms as F
import time
import threading

from utils_tasks.basic_utils import PlanningInput, PlanningOutput, find_usd_path, write_metrics, draw_box_with_text,adjust_usd_scale
from config_robots import *
from config_scenes import *
from config_tasks import *
from utils_tasks.client_utils import navigator_reset,nogoal_step
from utils_tasks.visualization_utils import VisualizationManager
from utils_tasks.tracking_utils import MPC_Controller
from utils_tasks.basic_utils import cpu_pointcloud_from_array

def update_occupancy(global_pcd, camera_int, current_pos, current_rot, robot_rgb, robot_depth):
    filter_rgb = torch.tensor(robot_rgb, device=robot_rgb.device)
    filter_depth = torch.tensor(robot_depth, device=robot_depth.device)
    filter_depth[filter_depth > 5.0] = 0
    points, colors = create_pointcloud_from_rgbd(
        camera_int,
        filter_depth,
        filter_rgb,
        position=current_pos,
        orientation=current_rot,
    )
    current_pcd = cpu_pointcloud_from_array(points.cpu().numpy(), colors.cpu().numpy())
    global_pcd = (global_pcd + current_pcd).voxel_down_sample(0.05)
    point_values = np.array(global_pcd.points)
    navigable_pcd = global_pcd.select_by_index(
        np.where(point_values[:, 2] < np.quantile(point_values[:, 2], 0.25) + 0.1)[0]
    )
    navigable_values = np.array(navigable_pcd.points)
    occupancy_dimension = np.ceil((navigable_values.max(axis=0) - navigable_values.min(axis=0)) / 0.1).astype(np.int32)
    occupancy_dimension[0] = max(occupancy_dimension[0], 1)
    occupancy_dimension[1] = max(occupancy_dimension[1], 1)
    occupancy_dimension[2] = max(occupancy_dimension[2], 1)
    occupancy_grid = np.zeros(occupancy_dimension)
    occupancy_index = np.floor((navigable_values - navigable_values.min(axis=0)) / 0.1).astype(np.int32)
    occupancy_grid[occupancy_index[:, 0], occupancy_index[:, 1], occupancy_index[:, 2]] = 1
    explore_area = occupancy_grid.sum() * 0.01
    return global_pcd, navigable_pcd, explore_area

planning_input = PlanningInput() 
planning_output = PlanningOutput()
input_lock = threading.Lock()
output_lock = threading.Lock()
stop_event = threading.Event()
vis_manager = [VisualizationManager(history_size=5) for i in range(args_cli.num_envs)]
mpc = None

def planning_thread(env, camera_intrinsic):
    global mpc
    """Thread function that continuously plans trajectories"""
    while not stop_event.is_set():
        try:
            # Get latest observations from shared state
            with input_lock:
                if planning_input.current_image is None or planning_input.current_depth is None or planning_input.camera_pos is None or planning_input.camera_rot is None:
                    time.sleep(0.01)
                    continue
                image = planning_input.current_image.copy()
                depth = planning_input.current_depth.copy()
                camera_pos = planning_input.camera_pos.copy()
                camera_rot = planning_input.camera_rot.copy()
            with output_lock:
                planning_output.is_planning = True
            
            # Start timing planning
            planning_start = time.time()
            trajectory_points_camera, all_trajectories_camera, all_values_camera = nogoal_step(image, depth,port=args_cli.port)
        
            # Transform trajectory from camera frame to world frame
            batch_optimal_points_world = []
            for idx in range(trajectory_points_camera.shape[0]):
                trajectory_points_world = []
                for i, point in enumerate(trajectory_points_camera[idx]):
                    if i < 0:
                        continue
                    point_local = np.array([point[0], point[1], 0.0])
                    point_world = camera_pos[idx] + camera_rot[idx] @ point_local
                    trajectory_points_world.append(point_world[:2])
                trajectory_points_world = np.array(trajectory_points_world)
                batch_optimal_points_world.append(trajectory_points_world)
                mpc = MPC_Controller(trajectory_points_world,
                                     desired_v=args_cli.speed,
                                     v_max=args_cli.speed,
                                     w_max=args_cli.speed)
            batch_optimal_points_world = np.array(batch_optimal_points_world)
           
            batch_all_points_world = []
            for idx in range(all_trajectories_camera.shape[0]):
                # Transform all trajectories
                all_trajectories_world = []
                for traj_camera in all_trajectories_camera[idx]:
                    traj_world = []
                    for point in traj_camera:
                        point_local = np.array([point[0], point[1], 0.0])
                        point_world = camera_pos[idx] + camera_rot[idx] @ point_local
                        traj_world.append(point_world[:2])
                    all_trajectories_world.append(np.array(traj_world))
                batch_all_points_world.append(all_trajectories_world)
            batch_all_points_world = np.array(batch_all_points_world)

            # Update shared state
            with output_lock:
                planning_output.trajectory_points_world = batch_optimal_points_world
                planning_output.all_trajectories_world = batch_all_points_world
                planning_output.all_values_camera = all_values_camera
                planning_output.is_planning = False
                planning_output.planning_error = None
            
            # Print planning timing
            planning_time = time.time() - planning_start
            # print(f"Planning time: {planning_time:.3f}s, Goal: [{goal[0]:.2f}, {goal[1]:.2f}, {goal[2]:.2f}]")
                
        except Exception as e:
            print(f"Planning error: {e}")
            with output_lock:
                planning_output.is_planning = False
                planning_output.planning_error = str(e)
        # Small sleep to prevent CPU overload
        time.sleep(0.1)

scene_path = os.path.join(args_cli.scene_dir,os.listdir(args_cli.scene_dir)[args_cli.scene_index]) + "/"
usd_path,init_path = find_usd_path(scene_path,task='pointgoal')
scene_config = ExplorationSceneCfg()
scene_config.num_envs = args_cli.num_envs
scene_config.env_spacing = 0.0
scene_config.terrain = BENCH_TERRAIN_CFG
scene_config.terrain.usd_path = usd_path
scene_config.robot = DINGO_CFG
scene_config.camera_sensor = DINGO_CameraCfg
scene_config.contact_sensor = DINGO_ContactCfg
scene_config.metric_sensor = DINGO_MetricCameraCfg
env_config = DingoExplorationCfg()
env_config.scene = scene_config
env_config.events.reset_pose.params = {"init_point_path":init_path, 
                                       'height_offset':0.1,
                                       'robot_visible': False,
                                       'light_enabled': False}
env = ManagerBasedRLEnv(env_config)
env = RslRlVecEnvWrapper(env)
adjust_usd_scale(scale=args_cli.scene_scale)
_,infos = env.reset()
camera_intrinsic = env.unwrapped.scene.sensors['camera_sensor'].data.intrinsic_matrices[0]

planning_thread_obj = threading.Thread(target=planning_thread, args=(env, camera_intrinsic))
planning_thread_obj.daemon = True
planning_thread_obj.start()

controller = DifferentialController(name="simple_control", 
                                    wheel_radius=DINGO_WHEEL_RADIUS,
                                    wheel_base=DINGO_WHEEL_BASE)
algo = navigator_reset(camera_intrinsic.cpu().numpy(),batch_size=scene_config.num_envs,stop_threshold=args_cli.stop_threshold,port=args_cli.port)
episode_num = args_cli.num_envs - 1

evaluation_metrics = []
save_dir = "./nogoal_%s_%s/%s/"%(algo,args_cli.scene_dir.split("/")[-1],scene_path.split("/")[-2])
os.makedirs(save_dir,exist_ok=True)
fps_writer = [imageio.get_writer(save_dir + "fps_%d.mp4"%i, fps=10) for i in range(scene_config.num_envs)]
global_pcds = [o3d.geometry.PointCloud() for i in range(scene_config.num_envs)]
navigable_pcds = [o3d.geometry.PointCloud() for i in range(scene_config.num_envs)]
explore_areas = np.zeros((scene_config.num_envs))
episode_steps = np.zeros((scene_config.num_envs,),dtype=np.int64)
trajectory_length = np.zeros((scene_config.num_envs))

while simulation_app.is_running():
    with torch.inference_mode():
        images = infos['observations']['rgb'].cpu().numpy()[:,:,:,0:3]
        depths = infos['observations']['depth'].cpu().numpy()[:,:,:]
     
        camera_pos = env.unwrapped.scene.sensors['camera_sensor'].data.pos_w.cpu().numpy()
        camera_rot_quat = env.unwrapped.scene.sensors['camera_sensor'].data.quat_w_world.cpu().numpy()
        camera_rot_quat = camera_rot_quat[:,[1, 2, 3, 0]]
        camera_rot = R.from_quat(camera_rot_quat).as_matrix()
        
        metric_camera_pos = env.unwrapped.scene.sensors["metric_sensor"].data.pos_w
        metric_camera_rot = env.unwrapped.scene.sensors["metric_sensor"].data.quat_w_ros
        metric_camera_int = env.unwrapped.scene.sensors["metric_sensor"].data.intrinsic_matrices
        metric_rgb = infos['observations']["metric_rgb"]
        metric_depth = infos['observations']["metric_depth"]
        for i in range(scene_config.num_envs):
            global_pcds[i], navigable_pcds[i], explore_areas[i] = update_occupancy(
                global_pcds[i],
                metric_camera_int[i],
                metric_camera_pos[i],
                metric_camera_rot[i],
                metric_rgb[i]/255.0,
                metric_depth[i],
            )
           
        with input_lock:
            planning_input.current_image = images.copy()
            planning_input.current_depth = depths.copy()
            planning_input.camera_pos = camera_pos.copy()
            planning_input.camera_rot = camera_rot.copy()

        # based on the current world trajectory 
        robot_vel = env.unwrapped.scene.articulations['robot'].data.root_lin_vel_w[0, :2].norm().cpu().numpy()
        robot_ang_vel = env.unwrapped.scene.articulations['robot'].data.root_ang_vel_w[0, 2].cpu().numpy()

        x0 = np.stack([camera_pos[:,0], camera_pos[:,1], np.arctan2(camera_rot[:,1,0], camera_rot[:,0,0]), [robot_vel], [robot_ang_vel]],axis=-1)
        current_trajectory = None
        current_all_trajectories = None
        current_all_values = None
        with output_lock:
            if planning_output.trajectory_points_world is not None:
                current_trajectory = planning_output.trajectory_points_world.copy() if planning_output.trajectory_points_world is not None else None
                current_all_trajectories = planning_output.all_trajectories_world.copy() if planning_output.all_trajectories_world is not None else None
                current_all_values = planning_output.all_values_camera.copy() if planning_output.all_values_camera is not None else None
        
        if current_trajectory is not None:
            control_start = time.time()
            action_list = []
            for i in range(args_cli.num_envs):
                vis_image = vis_manager[i].visualize_trajectory(
                    images[i], depths[i][:,:,None], camera_intrinsic.cpu().numpy(),
                    current_trajectory[i],
                    robot_pose=x0[i],
                    all_trajectories_points=current_all_trajectories[i],
                    all_trajectories_values=current_all_values[i]
                )
                if mpc is None:
                    continue
                t0 = time.time()
                opt_u_controls, opt_x_states = mpc.solve(x0[i,:3])
                print(f"solve mpc cost {time.time() - t0}")
                v, w = opt_u_controls[1, 0], opt_u_controls[1, 1]
                action = torch.tensor([v, w], device="cuda:0")
                action_cpu = action.cpu().numpy()
                joint_velocities = controller.forward(action_cpu).joint_velocities
                action_list.append(joint_velocities)
                
                try:
                    vis_image = draw_box_with_text(vis_image,0,0,430,50,"desired lin.:%.2f ang.:%.2f"%(v,w))
                    vis_image = draw_box_with_text(vis_image,0,50,430,50,"actual lin.:%.2f ang.:%.2f"%(robot_vel,robot_ang_vel))
                    if current_all_values is not None:
                        vis_image = draw_box_with_text(vis_image,0,770,430,50,"critic max:%.2f min:%.2f"%(np.max(current_all_values[i]), np.min(current_all_values[i])))
                    cv2.imwrite(f"frame_test.png", cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
                    fps_writer[i].append_data(vis_image)
                except:
                    pass
                
            action = torch.as_tensor(np.stack(action_list, axis=0),device="cuda:0")
            obs, rewards, dones, infos = env.step(action)
            # Get actual joint velocities from Isaac Sim
            actual_joint_velocities = env.unwrapped.scene.articulations['robot'].data.joint_vel[0, :2].cpu().numpy()
            desired_joint_velocities = env.unwrapped.scene.articulations['robot'].data.joint_vel_target[0, :2].cpu().numpy()
            trajectory_length += (infos['observations']['policy'][:,0] * env.unwrapped.step_dt).cpu().numpy()
            episode_steps += 1
        else:
            action = torch.zeros((args_cli.num_envs, 2), device="cuda:0")
            obs, rewards, dones, infos = env.step(action)
            episode_steps += 1
            print("No trajectory available, using zero action")
        
        for i in range(args_cli.num_envs):
            if dones[i] == True:
                episode_num += 1
                evaluation_metrics.append({'time':episode_steps[i] * env.env.step_dt,
                                           'area':explore_areas[i]})
                navigator_reset(env_id=i,port=args_cli.port)
                fps_writer[i].close()
                write_metrics(evaluation_metrics,save_dir+"metric.csv")
                fps_writer[i] = imageio.get_writer(save_dir + "fps_%d.mp4"%episode_num, fps=10)
                trajectory_length[i] = 0.0
                episode_steps[i] = 0
                global_pcds[i] = o3d.geometry.PointCloud()
                navigable_pcds[i] = o3d.geometry.PointCloud()
                explore_areas[i] = 0        
        
        if episode_num > args_cli.num_episodes:
            break
       
                
   

        
