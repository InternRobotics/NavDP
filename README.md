# NavDP System-1 Benchmark #
## 🏠 Introduction ##
This repository is a high-fidelity platform for benchmarking the visual navigation methods based on [IsaacSim](https://developer.nvidia.com/isaac/sim) and [IsaacLab](https://isaac-sim.github.io/IsaacLab/main/index.html). With realistic physics simulation and realistic scene assets, this repository aims to build an benchmark that can minimizing the sim-to-real gap in navigation system-1 evaluation.

![scenes](./asset_images/teasor.png)

### Highlights ###
- ⭐ Decoupled Framework between Navigation Approaches and Evaluation Process

The evaluation is accomplished by calling navigation method api with HTTP requests. By decoupling the implementation of navigation model with the evaluation process, it is much easier for users to evaluate the performance of novel navigation methods.


- ⭐ Fully Asynchronous Framework between Trajectory Planning and Following

We implement a MPC-based controller to constantly track the planned trajectory. With the asynchronous framework, the evaluation metrics become related to the navigation approaches' decision frequency which help align with the real-world navigation performance.

- ⭐ High-Quality Scene Asset for Evaluation

Our benchmark supports evaluation in diverse scene assets, including random cluttered environments, realistic home scenarios and commercial scenarios.

- ⭐ Support Image-Goal, Point-Goal and No-Goal Navigation Tasks

Our benchmark supports multiple navigation tasks, including no-goal exploration, point-goal navigation as well as image-goal navigation.

## 📋 Table of Contents
- [🏠 Introduction](#-introduction)
- [🌆 Prepare Scene Asset](#-prepare-scene-asset) 
- [🔧 Installation of Benchmark](#-installation-of-benchmark)
- [⚙️ Installation of Baseline Library](#️-installation-of-baseline-library)
- [💻 Running Basline as Server](#-running-basline-as-server)
- [🕹️ Running Teleoperation](#️-running-teleoperation)
- [📊 Running Evaluation](#-running-evaluation)
- [🔗 Citation](#-citation)
- [📄 License](#-license)
- [👏 Acknowledgements](#-acknowledgements)

## 🌆 Prepare Scene Asset ##
Please download the scene asset from [InternData-N1]() at HuggingFace.
The episodes information can be directly accessed in this repo. After downloading, please organize the structure as follows:
```bash
asset_scenes/
├── SkyTexture/
│   ├── belfast_sunset_puresky_4k.hdr
│   ├── citrus_orchard_road_puresky_4k.hdr
│   ├── ...
├── Materials/
│   ├── Carpet
│       ├── textures/
│       ├── Carpet_Woven.mdl
│       └── ...
│   ├── ...
├── cluttered_easy/
│   └── easy_0/
│       ├── cluttered-0.usd/
│       ├── imagegoal_start_goal_pairs.npy
│       └── pointgoal_start_goal_pairs.npy
│   ├── ...
├── cluttered_hard/
│   └── hard_0/
│       ├── cluttered-0.usd/
│       ├── imagegoal_start_goal_pairs.npy
│       └── pointgoal_start_goal_pairs.npy
│   ├── ...
├── internscenes_commercial/
│   └── MV4AFHQKTKJZ2AABAAAAADQ8_usd/
│       ├── models/
│       ├── Materials/
│       ├── metadata.json
│       ├── start_result_navigation.usd
│       └── pointgoal_start_goal_pairs.npy
│   ├── ...
├── internscene_home/
│   └── MV4AFHQKTKJZ2AABAAAAADQ8_usd/
│       ├── models/
│       ├── Materials/
│       ├── metadata.json
│       ├── start_result_navigation.usd
└──     └── pointgoal_start_goal_pairs.npy
```

| Category | Download Asset | Episodes |
|------|------|-------|
| SkyTexture | [Link]() | - |
| Materials  | [Link]() | - |
| Cluttered-Easy | [Link](./asset_scenes/cluttered_easy/) | [Episodes](./asset_scenes/cluttered_easy/) |
| Cluttered-Hard | [Link](./asset_scenes/cluttered_hard/) | [Episodes](./asset_scenes/cluttered_hard/) |
| InternScenes-Home |  [Link]() |  [Episodes](./asset_scenes/grutopia_home/) |
| InternScenes-Commercial | [Link]() | [Episodes](./asset_scenes/grutopia_commercial/) |

**Note: The textures and dataset are still waiting for uploading to HuggingFace**
## 🔧 Installation of Benchmark ##
Our framework is based on IsaacSim 4.2.0 and IsaacLab 1.2.0, you can follow the instructions to configure the conda environment.
```
# create the environment
conda create -n isaaclab python=3.10
conda activate isaaclab

# install IsaacSim 4.2
pip install --upgrade pip
pip install isaacsim==4.2.0.2 isaacsim-extscache-physics==4.2.0.2 isaacsim-extscache-kit==4.2.0.2 isaacsim-extscache-kit-sdk==4.2.0.2 --extra-index-url https://pypi.nvidia.com
# check the isaacsim installation
isaacsim omni.isaac.sim.python.kit

# install IsaacLab 1.2.0
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab/
git checkout tags/v1.2.0

# ignore the rsl-rl unavailable error
./isaaclab.sh -i 

# check the isaaclab installation
./isaaclab.sh -p source/standalone/tutorials/00_sim/create_empty.py
```
After preparing for the dependencies, please clone our project to get started.

```
git clone https://github.com/InternRobotics/NavDP.git
cd NavDP
git checkout navdp_benchmark
pip install -r requirements.txt
```

## ⚙️ Installation of Baseline Library ##
We collect the checkpoints for other navigation system-1 method from the corresponding respitory and organize their code to support the HTTP api calling for our benchmark. The links of paper, github codes as well as the pre-trained checkpoints are listed in the table below. Some of the baselines requires additional dependencies, and we provide the installation details below.
```
git clone https://github.com/InternRobotics/NavDP.git
cd NavDP
git checkout navdp_baseline
# you can rename the branch directory as nav_system1_baseline
mv ../NavDP ../nav_system1_baseline
```

| Baseline | Paper | Repo | Checkpoint | Support Tasks |
|------|------|-------|---------|----------|
| DD-PPO | [Arxiv](https://arxiv.org/abs/1911.00357) | [GitHub](https://github.com/facebookresearch/habitat-lab) | [Checkpoint](https://github.com/bdaiinstitute/vlfm/blob/main/data/pointnav_weights.pth)   | PointNav |
| iPlanner | [Arxiv](https://arxiv.org/abs/2302.11434)   | [GitHub](https://github.com/leggedrobotics/iPlanner) | [Checkpoint](https://drive.google.com/file/d/1UD11sSlOZlZhzij2gG_OmxbBN4WxVsO_/view?usp=share_link)   | PointNav |
| ViPlanner | [Arxiv](https://arxiv.org/abs/2310.00982)   | [GitHub](https://github.com/leggedrobotics/viplanner) | [Checkpoint](https://drive.google.com/file/d/1PY7XBkyIGESjdh1cMSiJgwwaIT0WaxIc/view?usp=sharing) [Mask2Former](https://drive.google.com/file/d/1DZoaLbXA1qPtg-gUKRUWS2rOH2tvDOOl/view?usp=sharing) | PointNav |
| GNM | [Arxiv](https://arxiv.org/abs/2210.03370)   | [GitHub](https://github.com/robodhruv/visualnav-transformer) | [Checkpoint](https://drive.google.com/drive/folders/1a9yWR2iooXFAqjQHetz263--4_2FFggg?usp=sharing) | ImageNav, NoGoal |
| ViNT | [Arxiv](https://arxiv.org/abs/2306.14846)   | [GitHub](https://github.com/robodhruv/visualnav-transformer) | [Checkpoint](https://drive.google.com/drive/folders/1a9yWR2iooXFAqjQHetz263--4_2FFggg?usp=sharing)| ImageNav, NoGoal |
| NoMad | [Arxiv](https://arxiv.org/abs/2310.07896)   | [GitHub](https://github.com/robodhruv/visualnav-transformer) | [Checkpoint](https://drive.google.com/drive/folders/1a9yWR2iooXFAqjQHetz263--4_2FFggg?usp=sharing) | ImageNav, NoGoal |
| NavDP | [Arxiv](https://arxiv.org/abs/2505.08712)  | [GitHub](https://github.com/OpenRobotLab/NavDP) | [Checkpoint](https://docs.google.com/forms/d/e/1FAIpQLSdl3RvajO5AohwWZL5C0yM-gkSqrNaLGp1OzN9oF24oNLfikw/viewform?usp=dialog) | PointNav, ImageNav, NoGoal |

### DD-PPO
To verify the performance of DD-PPO with continuous action space, we interpolate the predicted discrete actions {Stop, Forward, TurnLeft, TurnRight} into a trajectory. To play with the DD-PPO in our benchmark,
you need to install the habitat-lab and habitat-baselines. As Habitat only supports python <= 3.9, we recommand to create a new environment.
```
conda create -n habitat python=3.9 cmake=3.14.0
conda activate habitat
conda install habitat-sim withbullet -c conda-forge -c aihabitat
git clone --branch stable https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
pip install -e habitat-lab
pip install -e habitat-baselines
```

### iPlanner
No addition dependencies are required if you have configured the environment for running the benchmark.

### ViPlanner
For Viplanner, you need to install the mmcv and mmdet for Mask2Former. We recommand to create a new environment with torch 2.0.1 as backend.
```
pip install torch==2.0.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
pip install torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
pip install mmengine mmdet
pip install git+https://github.com/cocodataset/panopticapi.git
```

### GNM, ViNT and NoMad
To play with GNM, ViNT and NoMad, you need to install the following dependencies:
```
pip install efficientnet_pytorch==0.7.1
pip install diffusers==0.33.1
pip install git+https://github.com/real-stanford/diffusion_policy.git
```

## 💻 Running Basline as Server
To install the dependencies for different baseline method, please refer to [here](). For each pre-built baseline methods, each contains a server.py file, just simply run server python script with parsing the server port as well as the checkpoint path. You can download the checkpoints from the [baseline library table](). Taking NavDP as an example:
```
cd nav_system1_baseline/navdp/
python navdp_server.py --port 8888 --checkpoint ./checkpoints/navdp_checkpoint.ckpt
```
Then, the server will run at backend waiting for RGB-D observations and generate the preferred navigation trajectories.

## 🕹️ Running Teleoperation
For quickstart or debug with your novel navigation approach, we provide a teleoperation script that the robot move according to your teleoperation command while outputs the predicted trajectory for visualization. With a running server, the teleoperation code can be directly started with one-line command:
```
# if the running server support no-goal task
python teleop_nogoal_wheeled.py
# if the running server support point-goal task
python teleop_pointgoal_wheeled.py
# if the running server support image-goal task
python teleop_imagegoal_wheeled.py 
```
Then, you can use 'w','a','s','d' on the keyboard to control the linear and anguler speed.

## 📊 Running Evaluation
With a running server, it is simple to start the evaluation as:
```
# if the running server support no-goal task, Please Parse the Absolute Path of the ASSET_SCENE
python eval_nogoal_wheeled.py --port {PORT} --scene_dir {ASSET_SCENE} --scene_index {INDEX}
# if the running server support point-goal task, Please Parse the Absolute Path of the ASSET_SCENE
python eval_pointgoal_wheeled.py --port {PORT} --scene_dir {ASSET_SCENE} --scene_index {INDEX}
# if the running server support image-goal task, Please Parse the Absolute Path of the ASSET_SCENE
python eval_imagegoal_wheeled.py --port {PORT} --scene_dir {ASSET_SCENE} --scene_index {INDEX}
```
Please parse the port to match the server port, and the scene asset as well as the scene index to decide the evaluate scenario.
The evaluation results of the baseline methods will be released in a few days.


## 📄 License 
The open-sourced code are under the <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License </a><a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/80x15.png" /></a>.


## 🔗 Citation

If you find our work helpful, please cite:
```bibtex
@misc{navdp,
    title = {NavDP: Learning Sim-to-Real Navigation Diffusion Policy with Privileged Information Guidance},
    author = {Wenzhe Cai, Jiaqi Peng, Yuqiang Yang, Yujian Zhang, Meng Wei, Hanqing Wang, Yilun Chen, Tai Wang and Jiangmiao Pang},
    year = {2025},
    booktitle={arXiv},
}
```

## 👏 Acknowledgement
- [InternUtopia](https://github.com/OpenRobotLab/GRUtopia) (Previously `GRUtopia`): The closed-loop evaluation and GRScenes-100 data in this framework relies on the InternUtopia framework.
- [Diffusion Policy](https://github.com/real-stanford/diffusion_policy): Diffusion policy implementation.
- [ViPlanner](https://github.com/leggedrobotics/viplanner): ViPlanner implementation.
- [iPlanner](https://github.com/leggedrobotics/iPlanner): iPlanner implementation.
- [visualnav-transformer](https://github.com/robodhruv/visualnav-transformer): NoMad, ViNT, GNM implementation.