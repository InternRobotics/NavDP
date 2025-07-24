# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.
from pointnav_network import *
import argparse
from flask import Flask, request, jsonify
from PIL import Image
import cv2
import json

parser = argparse.ArgumentParser()
parser.add_argument("--port",type=int,default=8888)
parser.add_argument("--checkpoint",type=str,default="./checkpoints/pointnav_weights.pth")
parser.add_argument("--device",type=str,default="cuda:0")
args = parser.parse_known_args()[0]
policy = load_pointnav_policy(args.checkpoint)
policy.to(args.device)
last_action = None 
hidden_states = None 
mask = None 
intrinsic = None
batchsize = None

def process_image(image,target_size=224):
    # H,W,C = image.shape
    # prop = target_size/max(H,W)
    # image = cv2.resize(image,(-1,-1),fx=prop,fy=prop)
    # pad_width = max((target_size - image.shape[1])//2,0)
    # pad_height = max((target_size - image.shape[0])//2,0)
    # pad_image = np.pad(image,((pad_height,pad_height),(pad_width,pad_width),(0,0)),mode='constant',constant_values=0)
    # image = cv2.resize(pad_image,(target_size,target_size))
    # image = np.array(image,np.float32)/255.0
    return_image = cv2.resize(image,(target_size,target_size))
    return return_image
    
def process_depth(depth,target_size=224):
    # H,W = depth.shape[0:2]
    # prop = target_size/max(H,W)
    # depth = cv2.resize(depth,(-1,-1),fx=prop,fy=prop)
    # pad_width = max((target_size - depth.shape[1])//2,0)
    # pad_height = max((target_size - depth.shape[0])//2,0)
    # pad_depth = np.pad(depth,((pad_height,pad_height),(pad_width,pad_width)),mode='constant',constant_values=0)
    # pad_depth[pad_depth > 10.0] = 0
    # pad_depth[pad_depth < 0.1] = 0
    # depth = cv2.resize(pad_depth,(target_size,target_size))
    # depth = np.array(depth,np.float32)
    return_depth = cv2.resize(depth,(target_size,target_size))
    return return_depth[:,:,np.newaxis]

def process_goal(goal,range=5.0):
    return_goal = np.clip(goal,-range,range)
    return return_goal

app = Flask(__name__)
@app.route("/navigator_reset", methods=['POST'])
def ddppo_reset():
    global last_action,hidden_states,mask,intrinsic,batchsize
    intrinsic = np.array(request.get_json().get('intrinsic'))
    batchsize = np.array(request.get_json().get('batch_size'))
    last_action = torch.ones(batchsize,1,device=args.device, dtype=torch.long)
    hidden_states = torch.zeros(batchsize,4,512,device=args.device, dtype=torch.float32)
    mask = torch.zeros(batchsize,1,device=args.device, dtype=torch.bool)
    return jsonify({'algo':"ddppo"})

@app.route("/navigator_reset_env", methods=['POST'])
def ddppo_reset_env():
    global last_action,hidden_states,mask,intrinsic,batchsize
    env_id = int(request.get_json().get('env_id'))
    last_action[env_id] = torch.ones_like(last_action[env_id])
    hidden_states[env_id] = torch.zeros_like(hidden_states[env_id])
    mask[env_id] = torch.zeros_like(mask[env_id]) 
    return jsonify({'algo':"ddppo"})

@app.route("/pointgoal_step",methods=['POST'])
def ddppo_step_pointgoal():
    global last_action,hidden_states,mask,intrinsic,batchsize
    image_file = request.files['image']
    depth_file = request.files['depth']
    
    goal_data = json.loads(request.form.get('goal_data'))
    goal_x = np.array(goal_data['goal_x'])
    goal_y = np.array(goal_data['goal_y'])
    goal = np.stack((goal_x,goal_y),axis=1)
    goal = process_goal(goal)
    rho = np.linalg.norm(goal,axis=1)
    theta = np.arctan2(goal[:,1],goal[:,0])
    pointgoal_with_gps_compass = np.stack([rho, theta], axis=1)
   
    image = Image.open(image_file.stream)
    image = image.convert('RGB')
    image = np.asarray(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = image.reshape((batchsize, -1, image.shape[1], 3))
    
    depth = Image.open(depth_file.stream)
    depth = depth.convert('I')
    depth = np.asarray(depth)[:,:,np.newaxis]
    depth = depth.astype(np.float32)/10000.0
    depth = depth.reshape((batchsize, -1, depth.shape[1], 1))
    
    process_images = []
    process_depths = []
    for i in range(batchsize):
        process_images.append(process_image(image[i]))
        process_depths.append(process_depth(depth[i]))
    process_images = np.array(process_images)
    process_depths = np.array(process_depths)
    
    observations = {
        "rgb": torch.from_numpy(process_images).to(device=args.device, dtype=torch.float32)/255.0,
        "depth": torch.from_numpy(process_depths).to(device=args.device, dtype=torch.float32),
        "pointgoal_with_gps_compass": torch.from_numpy(pointgoal_with_gps_compass).to(device=args.device, dtype=torch.float32),
    }
    with torch.no_grad():
        outputs = policy.act(
            observations,
            hidden_states,
            last_action,
            mask,
        )

    last_action = outputs[0]
    hidden_states = outputs[1]
    navi_trajectories = np.array([np.linspace([0.0,0.0,0.0],[0.0,0.0,0.0],20), #Stop
                         np.linspace([0.0,0.0,0.0],[1.0,0.0,0.0],20),          #Forward
                         np.linspace([0.0,0.0,0.0],[0.0,1.0,1.0],20),          #TurnLeft
                         np.linspace([0.0,0.0,0.0],[0.0,-1.0,-1.0],20)])        #TurnRight
    trajectory = navi_trajectories[outputs[0][:,0].cpu().numpy().astype(np.int32)]
    all_trajectory = navi_trajectories[outputs[0][:,0].cpu().numpy().astype(np.int32)][:,None,:,:]
    all_values = np.zeros((batchsize,1))
    
    return jsonify({'trajectory': trajectory.tolist(),
                    'all_trajectory': all_trajectory.tolist(),
                    'all_values': all_values.tolist()}) 

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=args.port)