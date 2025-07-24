from PIL import Image
from flask import Flask, request, jsonify
from nomad_agent import NoMaDAgent  
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
parser.add_argument("--data_config",type=str,default="./configs/data_config.yaml")
parser.add_argument("--nomad_checkpoint",type=str,default="./checkpoints/nomad.pth")
parser.add_argument("--nomad_config",type=str,default="./configs/nomad.yaml")
parser.add_argument("--sample_num",type=int,default=16)
parser.add_argument("--device",type=str,default="cuda:0")
args = parser.parse_known_args()[0]

app = Flask(__name__)
nomad_navigator = None
nomad_fps_writer = None

@app.route("/navigator_reset",methods=['POST'])
def nomad_reset():
    global nomad_navigator,nomad_fps_writer
    intrinsic = np.array(request.get_json().get('intrinsic'))
    batchsize = np.array(request.get_json().get('batch_size'))
    if nomad_navigator is None:
        nomad_navigator = NoMaDAgent(intrinsic,
                                model_path=args.nomad_checkpoint,
                                model_config_path=args.nomad_config,
                                robot_config_path=args.robot_config,
                                data_config_path=args.data_config,
                                device=args.device)
        nomad_navigator.reset(batchsize)
    if nomad_fps_writer is None:
        format_time = datetime.datetime.fromtimestamp(time.time())
        format_time = format_time.strftime("%Y-%m-%d %H:%M:%S")
        nomad_fps_writer = imageio.get_writer("{}_fps_pointgoal.mp4".format(format_time),fps=7)
    else:
        nomad_fps_writer.close()
        format_time = datetime.datetime.fromtimestamp(time.time())
        format_time = format_time.strftime("%Y-%m-%d %H:%M:%S")
        nomad_fps_writer = imageio.get_writer("{}_fps_pointgoal.mp4".format(format_time),fps=7)
    return jsonify({"algo":"nomad"})

@app.route("/navigator_reset_env",methods=['POST'])
def nomad_reset_env():
    global nomad_navigator,nomad_fps_writer
    nomad_navigator.reset_env(int(request.get_json().get('env_id')))
    return jsonify({"algo":"nomad"})

@app.route("/nogoal_step",methods=['POST'])
def nomad_step_nogoal():
    global nomad_navigator,nomad_fps_writer
    image_file = request.files['image']
    depth_file = request.files['depth']
    
    image = Image.open(image_file.stream)
    image = image.convert('RGB')
    image = np.asarray(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = image.reshape((nomad_navigator.batch_size, -1, image.shape[1], 3))
    
    depth = Image.open(depth_file.stream)
    depth = depth.convert('I')
    depth = np.asarray(depth)[:,:,np.newaxis]
    depth = depth.astype(np.float32)/10000.0
    depth = depth.reshape((nomad_navigator.batch_size, -1, depth.shape[1], 1))
    
    _,trajectory,all_trajectory = nomad_navigator.step_nogoal(image,sample_num=args.sample_num)
    all_values = np.zeros((nomad_navigator.batch_size,all_trajectory.shape[1]))
    nomad_fps_writer.append_data(image.reshape(-1,image.shape[2],3))
    
    return jsonify({'trajectory': trajectory.cpu().numpy().tolist(),
                    'all_trajectory': all_trajectory.cpu().numpy().tolist(),
                    'all_values': all_values.tolist()})
    
@app.route("/imagegoal_step",methods=['POST'])
def nomad_step_imagegoal():
    global nomad_navigator,nomad_fps_writer
    image_file = request.files['image']
    depth_file = request.files['depth']
    goal_file = request.files['goal']
    
    image = Image.open(image_file.stream)
    image = image.convert('RGB')
    image = np.asarray(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = image.reshape((nomad_navigator.batch_size, -1, image.shape[1], 3))
    
    depth = Image.open(depth_file.stream)
    depth = depth.convert('I')
    depth = np.asarray(depth)[:,:,np.newaxis]
    depth = depth.astype(np.float32)/10000.0
    depth = depth.reshape((nomad_navigator.batch_size, -1, depth.shape[1], 1))
    
    goal = Image.open(goal_file.stream)
    goal = goal.convert('RGB')
    goal = np.asarray(goal)
    goal = cv2.cvtColor(goal, cv2.COLOR_RGB2BGR)
    goal = goal.reshape((nomad_navigator.batch_size, -1, goal.shape[1], 3))
    
    _,trajectory,all_trajectory = nomad_navigator.step_imagegoal(goal,image,sample_num=args.sample_num) #gnm_fps_writerm.step_pointgoal(image,depth,goal)
    all_values = np.zeros((nomad_navigator.batch_size,all_trajectory.shape[1]))
    nomad_fps_writer.append_data(image.reshape(-1,image.shape[2],3))
    
    return jsonify({'trajectory': trajectory.cpu().numpy().tolist(),
                    'all_trajectory': all_trajectory.cpu().numpy().tolist(),
                    'all_values': all_values.tolist()})

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=args.port)

        