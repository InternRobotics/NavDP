import argparse
from omni.isaac.lab.app import AppLauncher
from isaacsim import SimulationApp
parser = argparse.ArgumentParser(description="A script to run a car control simulation")
parser.add_argument(
    "--scene_dir", type=str, default="./asset_scenes/cluttered_easy")
parser.add_argument(
    "--scene_index", type=int, default=0)
parser.add_argument(
    "--stop_threshold", type=float, default=-2.0)
parser.add_argument(
    "--num_episodes", type=int, default=100)
parser.add_argument(
    "--port", type=int, default=8888)
args_cli = parser.parse_args()
app_launcher = AppLauncher(headless=False, enable_cameras=True)
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
import requests
import io
import json
import time
import threading
from queue import Queue
from typing import Optional, List, Tuple
from pynput import keyboard  # Replace keyboard with pynput.keyboard
from scipy.spatial.transform import Rotation as R
from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper
from wheeled_robots.controllers.differential_controller import DifferentialController

from utils_tasks.basic_utils import PlanningInput, PlanningOutput,find_usd_path
from utils_tasks.client_utils import navigator_reset,pointgoal_step
from config_robots import *
from config_scenes import *
from config_tasks import *
from utils_tasks.visualization_utils import VisualizationManager

# Global shared states
planning_input = PlanningInput()
planning_output = PlanningOutput()
input_lock = threading.Lock()
output_lock = threading.Lock()
stop_event = threading.Event()

def planning_thread(env, camera_intrinsic):
    """Thread function that continuously plans trajectories"""
    while not stop_event.is_set():
        try:
            # Get latest observations from shared state
            with input_lock:
                if planning_input.current_goal is None or planning_input.current_image is None or planning_input.current_depth is None or planning_input.camera_pos is None or planning_input.camera_rot is None:
                    time.sleep(0.01)
                    continue
                goal = planning_input.current_goal.copy()
                image = planning_input.current_image.copy()
                depth = planning_input.current_depth.copy()
                camera_pos = planning_input.camera_pos.copy()
                camera_rot = planning_input.camera_rot.copy()
            with output_lock:
                planning_output.is_planning = True
            
            # Start timing planning
            planning_start = time.time()
            trajectory_points_camera, all_trajectories_camera, all_values_camera = pointgoal_step(goal, image, depth, port=args_cli.port)
            print(trajectory_points_camera.shape,all_trajectories_camera.shape,all_values_camera.shape)
        
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
            batch_optimal_points_world = np.array(batch_optimal_points_world)
           
            batch_all_points_world = []
            for idx in range(all_trajectories_camera.shape[0]):
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

def main():
    scene_path = os.path.join(args_cli.scene_dir,os.listdir(args_cli.scene_dir)[args_cli.scene_index]) + "/"
    usd_path,init_path = find_usd_path(scene_path,'pointgoal')
    # Setup environment
    scene_config = PointNavSceneCfg()
    scene_config.num_envs = 1
    scene_config.env_spacing = 0.0
    scene_config.terrain = BENCH_TERRAIN_CFG
    scene_config.terrain.usd_path = usd_path
    scene_config.goal = GOAL_CFG
    scene_config.robot = DINGO_CFG
    scene_config.contact_sensor = DINGO_ContactCfg
    scene_config.camera_sensor = DINGO_CameraCfg
    
    env_config = DingoPointNavCfg()
    env_config.actions = DingoActionsCfg()
    env_config.scene = scene_config
    env_config.events.reset_pose.params = {"init_point_path": init_path,
                                           'height_offset': 0.25,
                                           'robot_visible': False,
                                           'light_enabled': True}
    env = ManagerBasedRLEnv(env_config)
    env = RslRlVecEnvWrapper(env)
    obs, infos = env.reset()
    camera_intrinsic = env.unwrapped.scene.sensors['camera_sensor'].data.intrinsic_matrices[0]
    controller = DifferentialController(name="simple_control", wheel_radius=DINGO_WHEEL_RADIUS, wheel_base=DINGO_WHEEL_BASE)
    algo = navigator_reset(camera_intrinsic.cpu().numpy(),batch_size=scene_config.num_envs,stop_threshold=args_cli.stop_threshold,port=args_cli.port)
    vis_manager = [VisualizationManager(history_size=5) for i in range(scene_config.num_envs)]

    # Start planning thread
    planning_thread_obj = threading.Thread(target=planning_thread, args=(env, camera_intrinsic))
    planning_thread_obj.daemon = True
    planning_thread_obj.start()

    # Initialize tracking variables
    episode_num = 0
    trajectory_length = np.zeros((scene_config.num_envs))
    save_dir = "./teleop_pointgoal_%s_%s/%s/"%(algo,args_cli.scene_dir.split("/")[-1],scene_path.split("/")[-2])
    os.makedirs(save_dir, exist_ok=True)
    euclidean = np.sqrt(np.square(infos['observations']['goal_pose'].cpu().numpy()[:,0:2]).sum(axis=-1))
    fps_writer = [imageio.get_writer(save_dir + "fps_%d.mp4"%i, fps=10) for i in range(scene_config.num_envs)]
    
    # Initialize dones
    dones = torch.zeros(1, dtype=torch.bool, device="cuda:0")
    linear_vel = 0.0
    angular_vel = 0.0
    max_linear_vel = 0.5  # Maximum linear velocity
    max_angular_vel = 0.5  # Maximum angular velocity
    vel_step = 0.1  # Velocity increment/decrement step

    # Initialize keyboard state
    key_state = {
        'w': False,
        's': False,
        'a': False,
        'd': False,
        'enter': False
    }

    def on_press(key):
        try:
            if key.char in key_state:
                key_state[key.char] = True
        except AttributeError:
            if key == keyboard.Key.enter:
                key_state['enter'] = True

    def on_release(key):
        try:
            if key.char in key_state:
                key_state[key.char] = False
        except AttributeError:
            if key == keyboard.Key.enter:
                key_state['enter'] = False

    # Start keyboard listener
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    # Main simulation loop
    while simulation_app.is_running():
        # Move CUDA operations to CPU before sharing
        goals = infos['observations']['goal_pose'].cpu().numpy()[:,0:2]
        images = infos['observations']['rgb'].cpu().numpy()[:,:,:,0:3]
        depths = infos['observations']['depth'].cpu().numpy()[:,:,:]
        # get all camera poses
        camera_pos = env.unwrapped.scene.sensors['camera_sensor'].data.pos_w.cpu().numpy()
        camera_rot_quat = env.unwrapped.scene.sensors['camera_sensor'].data.quat_w_world.cpu().numpy()
        camera_rot_quat = camera_rot_quat[:,[1, 2, 3, 0]]
        camera_rot = R.from_quat(camera_rot_quat).as_matrix()
        
        # Update shared state with latest observations
        with input_lock:
            planning_input.current_goal = goals.copy()
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
        
        # Handle keyboard input for control
        if key_state['w']:
            linear_vel = min(linear_vel + vel_step, max_linear_vel)
        elif key_state['s']:
            linear_vel = max(linear_vel - vel_step, -max_linear_vel)
        else:
            linear_vel = 0.0

        if key_state['a']:
            angular_vel = min(angular_vel + vel_step, max_angular_vel)
        elif key_state['d']:
            angular_vel = max(angular_vel - vel_step, -max_angular_vel)
        else:
            angular_vel = 0.0
        
        # Create visualization
        if current_trajectory is not None:
            for i in range(scene_config.num_envs):
                #try:
                vis_image = vis_manager[i].visualize_trajectory(
                    images[i], depths[i][:,:,None], camera_intrinsic.cpu().numpy(),
                    current_trajectory[i],
                    robot_pose=x0[i],
                    all_trajectories_points=current_all_trajectories[i],
                    all_trajectories_values=current_all_values[i]
                )
                cv2.imwrite(f"frame_test.png", cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
                fps_writer[i].append_data(vis_image)
            #except Exception as e:
            #    print(f"Visualization error: {e}")
        
        # Control using keyboard input
        action = torch.tensor([linear_vel, angular_vel], device="cuda:0")
        action_cpu = action.cpu().numpy()
        joint_velocities = controller.forward(action_cpu).joint_velocities
        action = torch.as_tensor(joint_velocities, device="cuda:0").unsqueeze(0)
        
        obs, rewards, dones, infos = env.step(action)
        
        # Get robot velocities
        robot_vel = env.unwrapped.scene.articulations['robot'].data.root_lin_vel_w[0, :2].norm().cpu().numpy()
        robot_ang_vel = env.unwrapped.scene.articulations['robot'].data.root_ang_vel_w[0, 2].cpu().numpy()
        
        # Get actual joint velocities from Isaac Sim
        actual_joint_velocities = env.unwrapped.scene.articulations['robot'].data.joint_vel[0, :4].cpu().numpy()
        desired_joint_velocities = env.unwrapped.scene.articulations['robot'].data.joint_vel_target[0, :4].cpu().numpy()

        print(f"actual joint vel:{actual_joint_velocities}  desired joint vel:{desired_joint_velocities}")
        trajectory_length += (infos['observations']['policy'][0,0] * env.unwrapped.step_dt).cpu().numpy()
        
        # Print control status
        print(f"Linear vel: {linear_vel:.3f}, Angular vel: {angular_vel:.3f}, Actual vel: {robot_vel:.3f} {robot_ang_vel:.2f}")
        
        # Check for Enter key press to reset
        if key_state['enter']:
            print("Reset triggered by Enter key press")
            episode_num += 1
            navigator_reset(env_id=i,port=args_cli.port)
            fps_writer[i].close()
            euclidean[i] = np.sqrt(np.square(infos['observations']['goal_pose'].cpu().numpy()[:,0:2]).sum(axis=-1))[i]
            fps_writer[i] = imageio.get_writer(save_dir + "fps_%d.mp4"%episode_num, fps=10)
            trajectory_length[i] = 0.0
        
        for i in range(scene_config.num_envs):
            if dones[i] == True:
                episode_num += 1
                navigator_reset(env_id=i,port=args_cli.port)
                fps_writer[i].close()
                euclidean[i] = np.sqrt(np.square(infos['observations']['goal_pose'].cpu().numpy()[:,0:2]).sum(axis=-1))[i]
                fps_writer[i] = imageio.get_writer(save_dir + "fps_%d.mp4"%episode_num, fps=10)
                trajectory_length[i] = 0.0

    # Cleanup
    stop_event.set()
    planning_thread_obj.join()
    listener.stop()
    fps_writer[0].close()

if __name__ == "__main__":
    main()