import numpy as np
import torch
from typing import List
from base_agent import NoMadBaseAgent
from nomad_model import NoMaDPolicy
from PIL import Image as PILImage
from torchvision import transforms
import torchvision.transforms.functional as TF
import yaml
import traj_opt

VISUALIZATION_IMAGE_SIZE = (160, 120)
IMAGE_ASPECT_RATIO = (4 / 3)
def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()
def from_numpy(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(array).float()
def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats["max"] - stats["min"]) + stats["min"]
    return data
def transform_images(pil_imgs: List[PILImage.Image], image_size: List[int], center_crop: bool = False) -> torch.Tensor:
    """Transforms a list of PIL image to a torch tensor."""
    transform_type = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    if type(pil_imgs) != list:
        pil_imgs = [pil_imgs]
    transf_imgs = []
    for pil_img in pil_imgs:
        w, h = pil_img.size
        if center_crop:
            if w > h:
                pil_img = TF.center_crop(pil_img, (h, int(h * IMAGE_ASPECT_RATIO)))  # crop to the right ratio
            else:
                pil_img = TF.center_crop(pil_img, (int(w / IMAGE_ASPECT_RATIO), w))
        pil_img = pil_img.resize(image_size)
        transf_img = transform_type(pil_img)
        transf_img = torch.unsqueeze(transf_img, 0)
        transf_imgs.append(transf_img)
    return torch.cat(transf_imgs, dim=1)



class NoMaDAgent(NoMadBaseAgent):
    def __init__(
        self,
        image_intrinsic,
        model_path: str,
        model_config_path: str,
        robot_config_path: str,
        data_config_path: str,
        device="cuda:0",
    ):
        super(NoMaDAgent, self).__init__(image_intrinsic, model_path, model_config_path, robot_config_path, device)
        self.nomad_former = NoMaDPolicy(self.cfg)
        self.nomad_former.to(self.device)
        self.nomad_former.model.load_state_dict(torch.load(self.model_path, map_location=self.device), strict=True)
        self.nomad_former.eval()
        self.traj_generate = traj_opt.TrajOpt()
        with open(data_config_path, "r") as f:
            data_config = yaml.safe_load(f)
        self.ACTION_STATS = {}
        for key in data_config["action_stats"]:
            self.ACTION_STATS[key] = np.array(data_config["action_stats"][key])
            
    def get_action(self,diffusion_output, action_stats):
        # diffusion_output: (B, 2*T+1, 1)
        # return: (B, T-1)
        device = diffusion_output.device
        ndeltas = diffusion_output
        ndeltas = ndeltas.reshape(ndeltas.shape[0], -1, 2)
        ndeltas = to_numpy(ndeltas)
        ndeltas = unnormalize_data(ndeltas, action_stats)
        actions = np.cumsum(ndeltas, axis=1)
        return from_numpy(actions).to(device)
    
    def step_nogoal(self, image, sample_num=8):
        with torch.no_grad():
            self.callback_obs(image)
            fake_goal = torch.randn((image.shape[0], 3, self.image_size[1], self.image_size[0])).to(self.device)
            input_image = [
                transform_images(imgs,self.image_size, center_crop=False).to(self.device) for imgs in self.memory_queue
            ]
            input_image = torch.concat(input_image, dim=0)
            mask = torch.ones(image.shape[0]).long().to(self.device) # ignore the goal
            output_actions = self.nomad_former.predict_nogoal_action(input_image, fake_goal, mask, sample_num=sample_num)
            output_actions = self.get_action(output_actions,self.ACTION_STATS)
            if self.normalize:
                output_actions[:,:,:2] *= self.MAX_V / self.RATE
            trajectory = self.traj_generate.TrajGeneratorFromPFreeRot(output_actions[:,:,0:2],step=0.1)
            
            reshape_trajectory = trajectory.reshape(self.batch_size,sample_num,trajectory.shape[1],-1)
            reshape_keypoints = output_actions.reshape(self.batch_size,sample_num,output_actions.shape[1],-1)
            return reshape_keypoints,reshape_trajectory[:,0],reshape_trajectory

    def step_imagegoal(self, goal_image, image, sample_num=8):
        with torch.no_grad():
            self.callback_obs(image)
            goal_image = [
                transform_images(PILImage.fromarray(g_img), self.image_size, center_crop=False).to(self.device) for g_img in goal_image
            ]
            goal_image = torch.concat(goal_image, dim=0)
            input_image = [
                transform_images(imgs,self.image_size, center_crop=False).to(self.device) for imgs in self.memory_queue
            ]
            input_image = torch.concat(input_image, dim=0)
            mask = torch.ones(1).long().to(self.device).repeat(self.batch_size)
            output_distances, obsgoal_cond = self.nomad_former.predict_imagegoal_distance(input_image, goal_image, mask)
            
            # [num_sample, pred_len, 2]
            output_actions = self.nomad_former.predict_imagegoal_action(obsgoal_cond,sample_num=sample_num)
            output_actions = self.get_action(output_actions,self.ACTION_STATS)
            if self.normalize:
                output_actions[:,:,:2] *= self.MAX_V / self.RATE
            trajectory = self.traj_generate.TrajGeneratorFromPFreeRot(output_actions[:,:,0:2],step=0.1)
            reshape_distances = (output_distances.reshape(self.batch_size,1,1,1) > 7.0).float()
            
            reshape_trajectory = trajectory.reshape(self.batch_size,sample_num,trajectory.shape[1],-1) * reshape_distances
            reshape_keypoints = output_actions.reshape(self.batch_size,sample_num,output_actions.shape[1],-1) * reshape_distances
            
            return reshape_keypoints,reshape_trajectory[:,0],reshape_trajectory 

if __name__ == "__main__":
    agent = NoMaDAgent(torch.ones((3,3)),
                    model_path="./checkpoints/nomad.pth",
                    model_config_path="./configs/nomad.yaml",
                    robot_config_path="./configs/robot_config.yaml",
                    data_config_path="./configs/data_config.yaml",
                    device='cuda:0')
    agent.reset(batch_size=2)
    image = np.random.rand(2,640,480,3).astype(np.uint8)
    goal = np.random.rand(2,640,480,3).astype(np.uint8)
    agent.step_nogoal(image)
    agent.step_imagegoal(goal,image)
