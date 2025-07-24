from PIL import Image
from flask import Flask, request, jsonify
from policy_agent import NavDP_Agent
import numpy as np
import cv2
import imageio
import time
import datetime
import json
import os

from PIL import Image, ImageDraw, ImageFont
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--port",type=int,default=8888)
parser.add_argument("--checkpoint",type=str,default="./checkpoints/cross-waic-pretrain-20.ckpt")
args = parser.parse_known_args()[0]

app = Flask(__name__)
navdp_navigator = None
navdp_fps_writer = None

@app.route("/navigator_reset",methods=['POST'])
def navdp_reset():
    global navdp_navigator,navdp_fps_writer
    intrinsic = np.array(request.get_json().get('intrinsic'))
    threshold = np.array(request.get_json().get('stop_threshold'))
    batchsize = np.array(request.get_json().get('batch_size'))
    if navdp_navigator is None:
        navdp_navigator = NavDP_Agent(intrinsic,
                                image_size=224,
                                memory_size=8,
                                predict_size=24,
                                temporal_depth=16,
                                heads=8,
                                token_dim=384,
                                navi_model=args.checkpoint,
                                device='cuda:0')
        navdp_navigator.reset(batchsize,threshold)
    else:
        navdp_navigator.reset(batchsize,threshold)

    if navdp_fps_writer is None:
        format_time = datetime.datetime.fromtimestamp(time.time())
        format_time = format_time.strftime("%Y-%m-%d %H:%M:%S")
        navdp_fps_writer = imageio.get_writer("{}_fps_pointgoal.mp4".format(format_time),fps=7)
    else:
        navdp_fps_writer.close()
        format_time = datetime.datetime.fromtimestamp(time.time())
        format_time = format_time.strftime("%Y-%m-%d %H:%M:%S")
        navdp_fps_writer = imageio.get_writer("{}_fps_pointgoal.mp4".format(format_time),fps=7)
    return jsonify({"algo":"navdp"})

@app.route("/navigator_reset_env",methods=['POST'])
def navdp_reset_env():
    global navdp_navigator
    navdp_navigator.reset_env(int(request.get_json().get('env_id')))
    return jsonify({"algo":"navdp"})

@app.route("/pointgoal_step",methods=['POST'])
def navdp_step_xy():
    global navdp_navigator,navdp_fps_writer
    start_time = time.time()
    image_file = request.files['image']
    depth_file = request.files['depth']
    goal_data = json.loads(request.form.get('goal_data'))
    goal_x = np.array(goal_data['goal_x'])
    goal_y = np.array(goal_data['goal_y'])
    goal = np.stack((goal_x,goal_y,np.ones_like(goal_x)),axis=1)
    batch_size = navdp_navigator.batch_size
    
    phase1_time = time.time()
    image = Image.open(image_file.stream)
    image = image.convert('RGB')
    image = np.asarray(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = image.reshape((batch_size, -1, image.shape[1], 3))
    
    depth = Image.open(depth_file.stream)
    depth = depth.convert('I')
    depth = np.asarray(depth)[:,:,np.newaxis]
    depth = depth.astype(np.float32)/10000.0
    depth = depth.reshape((batch_size, -1, depth.shape[1], 1))
    
    phase2_time = time.time()
    execute_trajectory, all_trajectory, all_values, trajectory_mask = navdp_navigator.step_pointgoal(goal,image,depth)
    phase3_time = time.time()
    navdp_fps_writer.append_data(trajectory_mask)
    phase4_time = time.time()
    print("phase1:%f, phase2:%f, phase3:%f, phase4:%f, all:%f"%(phase1_time - start_time, phase2_time - phase1_time, phase3_time - phase2_time, phase4_time-phase3_time, time.time() - start_time))

    return jsonify({'trajectory': execute_trajectory.tolist(),
                    'all_trajectory': all_trajectory.tolist(),
                    'all_values': all_values.tolist()})


@app.route("/pixelgoal_step",methods=['POST'])
def navdp_step_pixel():
    global navdp_navigator,navdp_fps_writer
    
    start_time = time.time()
    image_file = request.files['image']
    depth_file = request.files['depth']
    goal_data = json.loads(request.form.get('goal_data'))
    goal_x = np.array(goal_data['goal_x'])
    goal_y = np.array(goal_data['goal_y'])
    goal = np.stack((goal_x,goal_y),axis=1)
    batch_size = navdp_navigator.batch_size
    
    phase1_time = time.time()
    image = Image.open(image_file.stream)
    image = image.convert('RGB')
    image = np.asarray(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = image.reshape((batch_size, -1, image.shape[1], 3))
    
    depth = Image.open(depth_file.stream)
    depth = depth.convert('I')
    depth = np.asarray(depth)[:,:,np.newaxis]
    depth = depth.astype(np.float32)/10000.0
    depth = depth.reshape((batch_size, -1, depth.shape[1], 1))
    
    phase2_time = time.time()
    execute_trajectory, all_trajectory, all_values, trajectory_mask = navdp_navigator.step_pixelgoal(goal,image,depth)
    phase3_time = time.time()
    navdp_fps_writer.append_data(trajectory_mask)
    phase4_time = time.time()
    print("phase1:%f, phase2:%f, phase3:%f, phase4:%f, all:%f"%(phase1_time - start_time, phase2_time - phase1_time, phase3_time - phase2_time, phase4_time-phase3_time, time.time() - start_time))
    return jsonify({'trajectory': execute_trajectory.tolist(),
                    'all_trajectory': all_trajectory.tolist(),
                    'all_values': all_values.tolist()})

@app.route("/imagegoal_step",methods=['POST'])
def navdp_step_image():
    global navdp_navigator,navdp_fps_writer
    start_time = time.time()
    image_file = request.files['image']
    depth_file = request.files['depth']
    goal_file = request.files['goal']
    batch_size = navdp_navigator.batch_size
    
    phase1_time = time.time()
    image = Image.open(image_file.stream)
    image = image.convert('RGB')
    image = np.asarray(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite("image.jpg",image)
    image = image.reshape((batch_size, -1, image.shape[1], 3))
    
    goal = Image.open(goal_file.stream)
    goal = goal.convert('RGB')
    goal = np.asarray(goal)
    goal = cv2.cvtColor(goal, cv2.COLOR_RGB2BGR)
    cv2.imwrite("goal.jpg",goal)
    goal = goal.reshape((batch_size, -1, goal.shape[1], 3))
    
    depth = Image.open(depth_file.stream)
    depth = depth.convert('I')
    depth = np.asarray(depth)[:,:,np.newaxis]
    depth = depth.astype(np.float32)/10000.0
    depth = depth.reshape((batch_size, -1, depth.shape[1], 1))
    
    phase2_time = time.time()
    execute_trajectory, all_trajectory, all_values, trajectory_mask = navdp_navigator.step_imagegoal(goal,image,depth)
    phase3_time = time.time()
    navdp_fps_writer.append_data(trajectory_mask)
    phase4_time = time.time()
    print("phase1:%f, phase2:%f, phase3:%f, phase4:%f, all:%f"%(phase1_time - start_time, phase2_time - phase1_time, phase3_time - phase2_time, phase4_time-phase3_time, time.time() - start_time))
    return jsonify({'trajectory': execute_trajectory.tolist(),
                    'all_trajectory': all_trajectory.tolist(),
                    'all_values': all_values.tolist()})

@app.route("/nogoal_step",methods=['POST'])
def navdp_step_nogoal():
    global navdp_navigator,navdp_fps_writer
    start_time = time.time()
    image_file = request.files['image']
    depth_file = request.files['depth']
    batch_size = navdp_navigator.batch_size
    
    phase1_time = time.time()
    image = Image.open(image_file.stream)
    image = image.convert('RGB')
    image = np.asarray(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = image.reshape((batch_size, -1, image.shape[1], 3))
    
    depth = Image.open(depth_file.stream)
    depth = depth.convert('I')
    depth = np.asarray(depth)[:,:,np.newaxis]
    depth = depth.astype(np.float32)/10000.0
    depth = depth.reshape((batch_size, -1, depth.shape[1], 1))
    
    phase2_time = time.time()
    execute_trajectory, all_trajectory, all_values, trajectory_mask = navdp_navigator.step_nogoal(image,depth)
    phase3_time = time.time()
    navdp_fps_writer.append_data(trajectory_mask)
    phase4_time = time.time()
    print("phase1:%f, phase2:%f, phase3:%f, phase4:%f, all:%f"%(phase1_time - start_time, phase2_time - phase1_time, phase3_time - phase2_time, phase4_time-phase3_time, time.time() - start_time))
    return jsonify({'trajectory': execute_trajectory.tolist(),
                    'all_trajectory': all_trajectory.tolist(),
                    'all_values': all_values.tolist()})

@app.route("/navdp_step_ip_mixgoal",methods=['POST'])
def navdp_step_ip_mixgoal():
    global navdp_navigator,navdp_fps_writer
    start_time = time.time()
    image_file = request.files['image']
    depth_file = request.files['depth']
    batch_size = navdp_navigator.batch_size
    
    point_goal_data = json.loads(request.form.get('goal_data'))
    point_goal_x = np.array(point_goal_data['goal_x'])
    point_goal_y = np.array(point_goal_data['goal_y'])
    point_goal = np.stack((point_goal_x,point_goal_y,np.ones_like(point_goal_x)),axis=1)
    
    image_goal_file = request.files['image_goal']
    image_goal = Image.open(image_goal_file.stream)
    image_goal = image_goal.convert('RGB')
    image_goal = np.asarray(image_goal)
    image_goal = cv2.cvtColor(image_goal, cv2.COLOR_RGB2BGR)
    cv2.imwrite("goal.jpg",image_goal)
    image_goal = image_goal.reshape((batch_size, -1, image_goal.shape[1], 3))
    
    phase1_time = time.time()
    image = Image.open(image_file.stream)
    image = image.convert('RGB')
    image = np.asarray(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = image.reshape((batch_size, -1, image.shape[1], 3))
    
    depth = Image.open(depth_file.stream)
    depth = depth.convert('I')
    depth = np.asarray(depth)[:,:,np.newaxis]
    depth = depth.astype(np.float32)/10000.0
    depth = depth.reshape((batch_size, -1, depth.shape[1], 1))
    
    phase2_time = time.time()
    execute_trajectory, all_trajectory, all_values, trajectory_mask = navdp_navigator.step_point_image_goal(point_goal,image_goal,image,depth)
    phase3_time = time.time()
    navdp_fps_writer.append_data(trajectory_mask)
    phase4_time = time.time()
    print("phase1:%f, phase2:%f, phase3:%f, phase4:%f, all:%f"%(phase1_time - start_time, phase2_time - phase1_time, phase3_time - phase2_time, phase4_time-phase3_time, time.time() - start_time))
    return jsonify({'trajectory': execute_trajectory.tolist(),
                    'all_trajectory': all_trajectory.tolist(),
                    'all_values': all_values.tolist()})
    

if __name__ == "__main__":
    app.run(host='127.0.0.1',port=args.port)
