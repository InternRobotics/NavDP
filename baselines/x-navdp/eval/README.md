# X-NavDP Evaluation

This module evaluates X-NavDP point-goal navigation policies in Isaac Lab.

## Architecture

The evaluation system consists of two components:

1. **Policy Server** (`src/policy_server.py`): A Flask-based HTTP server that runs X-NavDP policy inference. It accepts RGB-D images and goals, and returns navigation trajectories.

2. **Evaluation Client** (`scripts/evaluate_pointgoal.py`): The main evaluation script that runs in Isaac Lab. It manages the simulation environment and communicates with the policy server.

## Directory Structure

```
eval/
├── README.md
├── config/
│   └── eval_pointgoal/        # Point-goal evaluation configuration files
│       ├── quadruped_clutter_easy.yaml
│       ├── quadruped_clutter_hard.yaml
│       ├── humanoid_internscene_home.yaml
│       └── wheeled_internscene_home.yaml
├── scripts/                  # Evaluation scripts
│   ├── evaluate_pointgoal.py
│   ├── start_policy_server.sh
│   └── run_evaluation.sh
└── src/                      # Source code
    ├── __init__.py
    ├── policy_server.py      # Flask server for policy inference
    ├── policy_agent.py      # High-level policy wrapper
    ├── policy_backbone.py    # CNN encoders (Depth Anything V2)
    ├── policy_network_embodiment.py  # Diffusion policy with embodiment modulation
    └── client_utils.py      # Client utilities for server communication
```

## Dependencies

This module requires:
- Isaac Sim + Isaac Lab
- PyTorch + diffusers
- Flask (for the policy server)
- All dependencies listed in the main `requirements.txt`

## Usage

### 1. Start the Policy Server

First, start the policy server in one terminal:

```bash
cd NavDP/baselines/x-navdp
bash eval/scripts/start_policy_server.sh \
    --checkpoint /path/to/navdp_model.ckpt \
    --embodiment quadruped \
    --device cuda:0
```

Supported embodiments:
- `wheeled`: Dingo robot
- `humanoid`: Unitree G1 robot
- `quadruped`: Unitree Go2 robot

### 2. Run Evaluation

In another terminal, run the evaluation:

```bash
cd NavDP/baselines/x-navdp
bash eval/scripts/run_evaluation.sh \
    --config_file eval/config/eval_pointgoal/quadruped_clutter_easy.yaml
```

### 3. View Results

Evaluation metrics are saved to:
- `metric.csv`: Success rate, SPL, and other metrics
- `fps_*.mp4`: Visualization videos for each episode

## Configuration

Point-goal configuration files in `eval/config/eval_pointgoal/` define:
- Environment settings (embodiment, task type, scene)
- Controller parameters
- Observation mappings
- MPC parameters

Home/commercial configs use the NavDP-style scene root at `data/scenes` and navigation metadata from `data/scenes/navigation_metadata/internscenes_home` or `data/scenes/navigation_metadata/internscenes_commercial`. Evaluation point-goal files are expected as `pointgoal_start_goal_pairs.npy` under each eval scene's `pointgoal_start_pair/<scene_name>/` metadata directory.

## Key Components

### Policy Server Routes

| Route | Method | Description |
|-------|--------|-------------|
| `/navigator_reset` | POST | Initialize navigator with camera intrinsics |
| `/navigator_reset_env` | POST | Reset a specific environment |
| `/pointgoal_step` | POST | Process point-goal navigation step |
| `/shutdown` | POST | Shutdown the server |

### Policy Agent Features

- **Trajectory Planning**: Uses diffusion model to generate navigation trajectories
- **Stuck Detection**: Detects when the robot is stuck and adjusts planning
- **Trajectory Guidance**: Uses previous trajectory execution for smooth navigation
- **Visualization**: Generates trajectory visualization masks

## License

See the X-NavDP [LICENSE](../LICENSE).
