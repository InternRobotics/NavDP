"""Batch MPC trajectory tracker built on acados."""

import casadi as ca
import numpy as np
import os
import scipy.linalg
import threading
from scipy.interpolate import interp1d

try:
    from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
except ModuleNotFoundError:
    AcadosOcp = AcadosOcpSolver = AcadosModel = None


def _require_acados_template():
    """Raise a clear error when the acados Python interface is unavailable."""
    if AcadosOcp is None or AcadosOcpSolver is None or AcadosModel is None:
        raise ModuleNotFoundError(
            "acados_template is required for BatchMPCController. Install acados and set "
            "ACADOS_SOURCE_DIR, or add <acados>/interfaces/acados_template to PYTHONPATH."
        )

_solver_init_lock = threading.Lock()
class MobilePlatformModel:
    """Simplified mobile platform model (only mobile base, no manipulator)"""
    def __init__(self):
        """Create the acados unicycle model with state ``x, y, theta``."""
        _require_acados_template()
        model = AcadosModel()
        # State variables: x, y, theta
        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        theta = ca.SX.sym('theta')
        states = ca.vertcat(x, y, theta)
        v = ca.SX.sym('v')
        w = ca.SX.sym('w')
        controls = ca.vertcat(v, w)
        # Kinematic model
        rhs = [v * ca.cos(theta), v * ca.sin(theta), w]
        # Create function
        f = ca.Function('f', [states, controls], [ca.vcat(rhs)])
        # Acados model
        x_dot = ca.SX.sym('x_dot', len(rhs))
        f_impl = x_dot - f(states, controls)
        model.f_expl_expr = f(states, controls)
        model.f_impl_expr = f_impl
        model.x = states
        model.xdot = x_dot
        model.u = controls
        model.p = []
        model.name = 'mobile_platform'
        self.model = model

class MPC_Controller_Fast:
    """Fast MPC controller based on Acados"""
    def __init__(self, N=15, desired_v=0.5, v_max=0.5, w_max=0.5, ref_gap=1, T=0.1,
                 ref_traj_length_m=2.0, ref_desired_v=0.5, min_desired_v=0.1, **_deprecated):
        """Configure and build a single acados MPC solver instance."""
        self.v_max = v_max
        self.w_max = w_max
        self.ref_traj_length_m = ref_traj_length_m
        self.ref_desired_v = ref_desired_v
        self.min_desired_v = min_desired_v

        self.N, self.desired_v, self.ref_gap, self.T = N, desired_v, ref_gap, T
        # Create model with unique name for each N
        platform_model = MobilePlatformModel()
        model = platform_model.model
        model.name = f'mobile_platform_N{N}'  # Unique name for each horizon

        nx = model.x.size()[0]  # 3
        self.nx = nx
        nu = model.u.size()[0]  # 2
        self.nu = nu
        ny = nx + nu  # 5

        # Create OCP
        ocp = AcadosOcp()
        ocp.model = model

        # Set prediction horizon (use new API)
        if hasattr(ocp.solver_options, 'N_horizon'):
            ocp.solver_options.N_horizon = self.N
        else:
            ocp.dims.N = self.N

        ocp.solver_options.tf = self.T * self.N

        # Cost function weights
        Q = np.diag([10.0, 10.0, 0.0])
        R = np.diag([0.05, 0.05])

        # Linear least squares cost
        ocp.cost.cost_type = 'LINEAR_LS'
        ocp.cost.cost_type_e = 'LINEAR_LS'
        ocp.cost.W = scipy.linalg.block_diag(Q, R)
        ocp.cost.W_e = Q

        ocp.cost.Vx = np.zeros((ny, nx))
        ocp.cost.Vx[:nx, :nx] = np.eye(nx)
        ocp.cost.Vu = np.zeros((ny, nu))
        ocp.cost.Vu[-nu:, -nu:] = np.eye(nu)
        ocp.cost.Vx_e = np.eye(nx)

        # Set control constraints
        ocp.constraints.lbu = np.array([-v_max, -w_max])
        ocp.constraints.ubu = np.array([v_max, w_max])
        ocp.constraints.idxbu = np.array([0, 1])

        # Set initial state (will be updated in solve)
        x_ref = np.zeros(nx)
        u_ref = np.zeros(nu)
        ocp.constraints.x0 = x_ref
        ocp.cost.yref = np.concatenate((x_ref, u_ref))
        ocp.cost.yref_e = x_ref

        # Solver options (match original code)
        ocp.solver_options.nlp_solver_max_iter = 100
        ocp.solver_options.qp_solver_iter_max = 100
        ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'  # Use full condensing like original
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.print_level = 0
        ocp.solver_options.nlp_solver_type = 'SQP'

        # Set acados path
        acados_source_path = os.environ.get('ACADOS_SOURCE_DIR', '')
        if acados_source_path:
            # acados_include_path is read-only and set automatically, don't set it
            # Only set acados_lib_path through code_gen_opts if the path exists
            acados_lib_path = os.path.join(acados_source_path, 'lib')
            link_libs_json = os.path.join(acados_lib_path, 'link_libs.json')
            # Only set acados_lib_path if the path exists and contains link_libs.json
            # Otherwise, let acados use its default auto-detected path
            if os.path.exists(link_libs_json):
                ocp.code_gen_opts.acados_lib_path = acados_lib_path
        # Create solver (unique for each N value)
        # Use rank-specific dir to avoid JSON corruption when multiple DDP ranks generate simultaneously
        rank = os.environ.get('RANK', os.environ.get('LOCAL_RANK', '0'))
        codegen_root = os.environ.get(
            "X_NAVDP_MPC_CODEGEN_DIR",
            os.environ.get(
                "NAVRL_MPC_CODEGEN_DIR",
                os.path.join(os.path.expanduser("~"), ".x-navdp", "c_generated_code"),
            ),
        )
        code_gen_dir = os.path.join(codegen_root, f"rank_{rank}")
        os.makedirs(code_gen_dir, exist_ok=True)
        json_file = os.path.join(code_gen_dir, f'{model.name}_acados_ocp.json')
        # Set code export directory to ensure files are generated in the correct location
        ocp.code_gen_opts.code_export_directory = code_gen_dir
        # Thread-safe check and creation of solver
        with _solver_init_lock:
            # Check if code already exists to avoid recompilation in parallel scenarios
            generate_code = True
            so_file = f'libacados_ocp_solver_{model.name}.so'
            if os.path.exists(code_gen_dir) and os.path.exists(os.path.join(code_gen_dir, so_file)):
                generate_code = False
                print(f"  [Thread {threading.current_thread().name}] Reusing precompiled solver")
            else:
                print(f"  [Thread {threading.current_thread().name}] Generating and compiling solver...")
            self.solver = AcadosOcpSolver(ocp, json_file=json_file, generate=generate_code, build=generate_code)
        self.last_opt_x_states = None
        self.last_opt_u_controls = None

    def make_ref_denser(self, ref_traj, ratio=50):
        """Make reference trajectory denser"""
        x_orig = np.arange(len(ref_traj))
        new_x = np.linspace(0, len(ref_traj) - 1, num=len(ref_traj) * ratio)
        interp_func_x = interp1d(x_orig, ref_traj[:, 0], kind='linear')
        interp_func_y = interp1d(x_orig, ref_traj[:, 1], kind='linear')
        uniform_x = interp_func_x(new_x)
        uniform_y = interp_func_y(new_x)
        ref_traj = np.stack((uniform_x, uniform_y), axis=1)
        return ref_traj

    def solve(self, x00 = np.zeros((3,))):
        """Solve MPC problem"""
        ref_traj = self.find_reference_traj(x00, self.ref_traj)
        # Add yaw angle (set to 0)
        ref_traj = np.concatenate((ref_traj, np.zeros((ref_traj.shape[0], 1))), axis=1)

        # Warm start using previous solution (shift by one step)
        if self.last_opt_x_states is not None and self.last_opt_u_controls is not None:
            # Shift previous solution and use as initial guess
            for i in range(self.N):
                if i < self.N - 1:
                    self.solver.set(i, 'x', self.last_opt_x_states[i + 1])
                    self.solver.set(i, 'u', self.last_opt_u_controls[i + 1] if i + 1 < self.N else self.last_opt_u_controls[-1])
                else:
                    self.solver.set(i, 'x', self.last_opt_x_states[-1])
                    self.solver.set(i, 'u', self.last_opt_u_controls[-1])
            self.solver.set(self.N, 'x', self.last_opt_x_states[-1])
        else:
            # Set initial state constraint
            for i in range(self.N):
                self.solver.set(i, 'x', x00)
                self.solver.set(i, 'u', np.array([1, 1]) * 0.001)
        # Set reference trajectory for intermediate nodes
        # Use linear interpolation to ensure smooth reference for all nodes
        u_ref = np.zeros(self.nu)
        for i in range(self.N):
            # Calculate interpolated reference for each node
            idx_float = i / self.ref_gap
            idx_low = min(int(np.floor(idx_float)), self.ref_traj_len - 1)
            idx_high = min(int(np.ceil(idx_float)), self.ref_traj_len - 1)
            alpha = idx_float - idx_low

            if idx_low == idx_high:
                x_ref = ref_traj[idx_low, :]
            else:
                x_ref = (1 - alpha) * ref_traj[idx_low, :] + alpha * ref_traj[idx_high, :]

            yref = np.concatenate((x_ref, u_ref))
            self.solver.set(i, 'yref', yref)

        # Set terminal reference (only state, 3-dim)
        terminal_ref = ref_traj[-1, :].flatten()
        self.solver.set(self.N, 'yref', terminal_ref)

        # Set initial state constraint
        self.solver.set(0, "lbx", x00)
        self.solver.set(0, "ubx", x00)

        # Solve
        status = self.solver.solve()

        if status != 0:
            print(f"WARNING: Acados solver returned status {status}")

        # Extract solution
        x_opt = np.array([self.solver.get(i, 'x') for i in range(self.N + 1)])
        u_opt = np.array([self.solver.get(i, 'u') for i in range(self.N)])

        self.last_opt_x_states = x_opt
        self.last_opt_u_controls = u_opt

        return u_opt, x_opt

    def reset(self, global_ref_traj):
        """Reset solver"""
        self.last_opt_x_states = None
        self.last_opt_u_controls = None
        self.ref_traj = self.make_ref_denser(global_ref_traj)
        self.ref_traj_len = self.N // self.ref_gap + 1

        traj_length_m = self.compute_trajectory_length(global_ref_traj)
        desired_v_length = self.desired_v_from_trajectory_length(traj_length_m)

        traj_curvature = self.calculate_curvature(global_ref_traj)
        max_curvature = np.max(traj_curvature[:12])
        desired_v_curvature = self.adaptive_desired_velocity(max_curvature)

        self.desired_v = min(desired_v_length, desired_v_curvature)

        return max_curvature

    def calculate_curvature(self, traj):
        """Estimate smoothed planar curvature along a reference trajectory."""

        dx = np.gradient(traj[:, 0])
        dy = np.gradient(traj[:, 1])
        dy[0] = 0.0

        ddx = np.gradient(dx)
        ddy = np.gradient(dy)

        numerator = np.abs(dx * ddy - dy * ddx)
        denominator = (dx**2 + dy**2)**1.5
        denominator[denominator < 1e-6] = 1e-6

        curvature = numerator / denominator
        curvature = np.convolve(curvature, np.ones(3)/3, mode='same')

        return curvature

    def adaptive_desired_velocity(self, curvature):
        """Choose desired velocity from curvature and velocity bounds."""
        k = max(curvature, 1e-6)
        v_max_constraint = self.v_max
        v_curvature_constraint = self.w_max / k
        desired_v = min(v_max_constraint, v_curvature_constraint)
        desired_v = max(self.min_desired_v, desired_v)

        return desired_v

    @staticmethod
    def compute_trajectory_length(traj):
        """总轨迹弧长 sum(sqrt(Δx² + Δy²))；traj 首点已为原点。"""
        if traj.shape[0] < 2:
            return 0.0
        pts = traj[:, :2]
        return float(np.sum(np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))))

    def desired_v_from_trajectory_length(self, length_m):
        """Scale desired velocity down for short reference trajectories."""
        length_m = max(float(length_m), 1e-6)
        scale = min(length_m / self.ref_traj_length_m, 1.0)
        v = self.ref_desired_v * scale
        return max(self.min_desired_v, min(self.v_max, v))

    def find_reference_traj(self, x0, global_planed_traj, lookahead_points=10):
        """Find reference trajectory points

        Args:
            x0: current state [x, y, theta]
            global_planed_traj: global trajectory
            lookahead_points: number of points to look ahead from nearest point
        """
        ref_traj_pts = []
        # Find nearest point
        nearest_idx = np.argmin(np.linalg.norm(global_planed_traj - x0[:2].reshape((1, 2)), axis=1))

        # Look ahead from the nearest point
        start_idx = min(nearest_idx + lookahead_points, len(global_planed_traj) - 1)
        desire_arc_length = self.desired_v * self.ref_gap * self.T
        cum_dist = np.cumsum(np.linalg.norm(np.diff(global_planed_traj, axis=0), axis=1))

        # Select reference points from start_idx
        for i in range(start_idx, len(global_planed_traj) - 1):
            if cum_dist[i] - cum_dist[start_idx] >= desire_arc_length * len(ref_traj_pts):
                ref_traj_pts.append(global_planed_traj[i, :])
                if len(ref_traj_pts) == self.ref_traj_len:
                    break

        # Fill with terminal point if reference trajectory is not long enough
        while len(ref_traj_pts) < self.ref_traj_len:
            ref_traj_pts.append(global_planed_traj[-1, :])
        return np.array(ref_traj_pts)

class BatchMPCController:
    """Batch wrapper that reuses one MPC solver across reference trajectories."""
    def __init__(self, batch=1, N=15, desired_v=0.5, v_max=0.5, w_max=0.5, ref_gap=1, T=0.1,
                 ref_traj_length_m=2.0, ref_desired_v=0.5, min_desired_v=0.05, **_deprecated):
        """Create the reusable MPC controller used for batched training plans."""
        self.controller = MPC_Controller_Fast(
            N=N,
            desired_v=desired_v,
            v_max=v_max,
            w_max=w_max,
            ref_gap=ref_gap,
            T=T,
            ref_traj_length_m=ref_traj_length_m,
            ref_desired_v=ref_desired_v,
            min_desired_v=min_desired_v,
        )
    def solve(self,batch_ref_trajectories):
        """Solve MPC for each reference trajectory in the batch."""
        opt_us = []
        opt_xs = []
        real_desired_v = []
        max_traj_curvatures = []
        batchsize = batch_ref_trajectories.shape[0]
        for i in range(batchsize):
            max_traj_curvature = self.controller.reset(batch_ref_trajectories[i])
            opt_u, opt_x = self.controller.solve()
            opt_us.append(opt_u)
            opt_xs.append(opt_x)
            real_desired_v.append(self.controller.desired_v)
            max_traj_curvatures.append(max_traj_curvature)
        opt_us = np.stack(opt_us,axis=0)
        opt_xs = np.stack(opt_xs,axis=0)
        real_desired_v = np.stack(real_desired_v,axis=0)
        max_traj_curvatures = np.stack(max_traj_curvatures,axis=0)
        return opt_us, opt_xs, real_desired_v, max_traj_curvatures
