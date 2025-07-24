from PIL import Image
from flask import Flask, request, jsonify
from gnm_agent import GNMAgent  
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
parser.add_argument("--robot_config",type=str,default="./configs/robot_config.yaml")
parser.add_argument("--gnm_checkpoint",type=str,default="./checkpoints/gnm.pth")
parser.add_argument("--gnm_config",type=str,default="./configs/gnm.yaml")
parser.add_argument("--device",type=str,default="cuda:0")
args = parser.parse_known_args()[0]

app = Flask(__name__)
gnm_navigator = None
gnm_fps_writer = None

@app.route("/navigator_reset",methods=['POST'])
def gnm_reset():
    global gnm_navigator,gnm_fps_writer
    intrinsic = np.array(request.get_json().get('intrinsic'))
    batchsize = np.array(request.get_json().get('batch_size'))
    if gnm_navigator is None:
        gnm_navigator = GNMAgent(intrinsic,
                                model_path=args.gnm_checkpoint,
                                model_config_path=args.gnm_config,
                                robot_config_path=args.robot_config,
                                device=args.device)
        gnm_navigator.reset(batchsize)
        
    if gnm_fps_writer is None:
        format_time = datetime.datetime.fromtimestamp(time.time())
        format_time = format_time.strftime("%Y-%m-%d %H:%M:%S")
        gnm_fps_writer = imageio.get_writer("{}_fps_pointgoal.mp4".format(format_time),fps=7)
    else:
        gnm_fps_writer.close()
        format_time = datetime.datetime.fromtimestamp(time.time())
        format_time = format_time.strftime("%Y-%m-%d %H:%M:%S")
        gnm_fps_writer = imageio.get_writer("{}_fps_pointgoal.mp4".format(format_time),fps=7)
    return jsonify({"algo":"gnm"})

@app.route("/navigator_reset_env",methods=['POST'])
def gnm_reset_env():
    global gnm_navigator,gnm_fps_writer
    gnm_navigator.reset_env(int(request.get_json().get('env_id')))
    return jsonify({"algo":"gnm"})

@app.route("/nogoal_step",methods=['POST'])
def gnm_step_nogoal():
    global gnm_navigator,gnm_fps_writer
    image_file = request.files['image']
    depth_file = request.files['depth']
    
    image = Image.open(image_file.stream)
    image = image.convert('RGB')
    image = np.asarray(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = image.reshape((gnm_navigator.batch_size, -1, image.shape[1], 3))
    
    depth = Image.open(depth_file.stream)
    depth = depth.convert('I')
    depth = np.asarray(depth)[:,:,np.newaxis]
    depth = depth.astype(np.float32)/10000.0
    depth = depth.reshape((gnm_navigator.batch_size, -1, depth.shape[1], 1))
    
    _,trajectory = gnm_navigator.step_nogoal(image) #gnm_fps_writerm.step_pointgoal(image,depth,goal)
    all_values = np.zeros((gnm_navigator.batch_size,1))
    gnm_fps_writer.append_data(image.reshape(-1,image.shape[2],3))
    
    return jsonify({'trajectory': trajectory.cpu().numpy().tolist(),
                    'all_trajectory': trajectory.cpu().numpy()[None,:,:,:].tolist(),
                    'all_values': all_values.tolist()})
    
@app.route("/imagegoal_step",methods=['POST'])
def gnm_step_imagegoal():
    global gnm_navigator,gnm_fps_writer
    image_file = request.files['image']
    depth_file = request.files['depth']
    goal_file = request.files['goal']
    
    image = Image.open(image_file.stream)
    image = image.convert('RGB')
    image = np.asarray(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = image.reshape((gnm_navigator.batch_size, -1, image.shape[1], 3))
    
    depth = Image.open(depth_file.stream)
    depth = depth.convert('I')
    depth = np.asarray(depth)[:,:,np.newaxis]
    depth = depth.astype(np.float32)/10000.0
    depth = depth.reshape((gnm_navigator.batch_size, -1, depth.shape[1], 1))
    
    goal = Image.open(goal_file.stream)
    goal = goal.convert('RGB')
    goal = np.asarray(goal)
    goal = cv2.cvtColor(goal, cv2.COLOR_RGB2BGR)
    goal = goal.reshape((gnm_navigator.batch_size, -1, goal.shape[1], 3))
    
    _,trajectory = gnm_navigator.step_imagegoal(goal,image) #gnm_fps_writerm.step_pointgoal(image,depth,goal)
    all_values = np.zeros((gnm_navigator.batch_size,1))
    gnm_fps_writer.append_data(image.reshape(-1,image.shape[2],3))
    
    return jsonify({'trajectory': trajectory.cpu().numpy().tolist(),
                    'all_trajectory': trajectory.cpu().numpy()[None,:,:,:].tolist(),
                    'all_values': all_values.tolist()})

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=args.port)

        