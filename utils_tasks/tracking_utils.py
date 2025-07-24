import casadi as ca
import numpy as np
import time
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from scipy.interpolate import interp1d
from typing import Optional, List, Tuple
from dataclasses import dataclass
from queue import Queue

@dataclass
class PlanningInput:
    current_goal: Optional[np.ndarray] = None
    current_image: Optional[np.ndarray] = None
    current_depth: Optional[np.ndarray] = None
    camera_pos: Optional[np.ndarray] = None
    camera_rot: Optional[np.ndarray] = None

@dataclass
class PlanningOutput:
    trajectory_points_world: Optional[np.ndarray] = None
    all_trajectories_world: Optional[List[np.ndarray]] = None
    all_values_camera: Optional[np.ndarray] = None
    is_planning: bool = False
    planning_error: Optional[str] = None

class MPC_Controller:
    def __init__(self, global_planed_traj, N = 15, desired_v = 0.5, v_max = 0.5, w_max = 0.5, ref_gap = 3):
        self.N, self.desired_v, self.ref_gap, self.T = N, desired_v, ref_gap, 0.1
        
        self.ref_traj = self.make_ref_denser(global_planed_traj)
        self.ref_traj_len = N // ref_gap + 1

        # setup mpc problem
        opti = ca.Opti()
        opt_controls = opti.variable(N, 2)
        v, w = opt_controls[:, 0], opt_controls[:, 1]

        opt_states = opti.variable(N+1, 3)
        x, y, theta = opt_states[:, 0], opt_states[:, 1], opt_states[:, 2]

        # parameters 
        opt_x0 = opti.parameter(3)
        opt_xs = opti.parameter(3 * self.ref_traj_len) # the intermidia state may also be the parameter

        # system dynamics for mobile manipulator
        f = lambda x_, u_: ca.vertcat(*[u_[0]*ca.cos(x_[2]), u_[0]*ca.sin(x_[2]), u_[1]])

        # init_condition
        opti.subject_to(opt_states[0, :] == opt_x0.T)
        for i in range(N):
            x_next = opt_states[i, :] + f(opt_states[i, :], opt_controls[i, :]).T*self.T
            opti.subject_to(opt_states[i+1, :]==x_next)

        # define the cost function
        Q = np.diag([10.0,10.0,0.0])
        R = np.diag([0.02,0.15])
        obj = 0 
        for i in range(N):
            obj = obj +ca.mtimes([opt_controls[i, :], R, opt_controls[i, :].T])
            if i % ref_gap == 0:
                nn = i // ref_gap
                obj = obj + ca.mtimes([(opt_states[i, :]-opt_xs[nn*3:nn*3+3].T), Q, (opt_states[i, :]-opt_xs[nn*3:nn*3+3].T).T])
        opti.minimize(obj)

        # boundrary and control conditions
        opti.subject_to(opti.bounded(0.0, v, v_max))
        opti.subject_to(opti.bounded(-w_max, w, w_max))
        
        opts_setting = {'ipopt.max_iter':100, 'ipopt.print_level':0, 'print_time':0, 'ipopt.acceptable_tol':1e-8, 'ipopt.acceptable_obj_change_tol':1e-6}
        opti.solver('ipopt', opts_setting)
        
        self.opti = opti
        self.opt_xs = opt_xs
        self.opt_x0 = opt_x0
        self.opt_controls = opt_controls
        self.opt_states = opt_states
        self.last_opt_x_states = None
        self.last_opt_u_controls = None
    def make_ref_denser(self, ref_traj, ratio = 50):
        x_orig = np.arange(len(ref_traj))
        new_x = np.linspace(0, len(ref_traj) - 1, num=len(ref_traj) * ratio)
        interp_func_x = interp1d(x_orig, ref_traj[:, 0], kind='linear')
        interp_func_y = interp1d(x_orig, ref_traj[:, 1], kind='linear')
        uniform_x = interp_func_x(new_x)
        uniform_y = interp_func_y(new_x)
        ref_traj = np.stack((uniform_x, uniform_y), axis=1)
        return ref_traj
    
    def solve(self, x00):
        ref_traj = self.find_reference_traj(x00, self.ref_traj)
        # fake a yaw angle
        ref_traj = np.concatenate((ref_traj, np.zeros((ref_traj.shape[0], 1))), axis=1).reshape(-1, 1)
        self.opti.set_value(self.opt_xs, ref_traj.reshape(-1, 1)) 
        u0 = np.zeros((self.N, 2)) if self.last_opt_u_controls is None else self.last_opt_u_controls
        x0 = np.zeros((self.N+1, 3)) if self.last_opt_x_states is None else self.last_opt_x_states
        self.opti.set_value(self.opt_x0, x00)
        self.opti.set_initial(self.opt_controls, u0)
        self.opti.set_initial(self.opt_states, x0)
        sol = self.opti.solve()
        self.last_opt_u_controls = sol.value(self.opt_controls)
        self.last_opt_x_states = sol.value(self.opt_states)

        return self.last_opt_u_controls, self.last_opt_x_states
    def reset(self):
        self.last_opt_x_states = None
        self.last_opt_u_controls = None
        
    def find_reference_traj(self, x0, global_planed_traj):
        ref_traj_pts = []
        # find the nearest point in global_planed_traj
        nearest_idx = np.argmin(np.linalg.norm(global_planed_traj - x0[:2].reshape((1, 2)), axis=1))
        desire_arc_length = self.desired_v * self.ref_gap * self.T 
        cum_dist = np.cumsum(np.linalg.norm(np.diff(global_planed_traj, axis=0), axis=1))

        # select the reference points from the nearest point to the end of global_planed_traj
        for i in range(nearest_idx, len(global_planed_traj) - 1):
            if cum_dist[i] - cum_dist[nearest_idx] >= desire_arc_length * len(ref_traj_pts):
                ref_traj_pts.append(global_planed_traj[i, :])
                if len(ref_traj_pts) == self.ref_traj_len:
                    break
        # if the target is reached before the reference trajectory is complete, add the last point of global_planed_traj 
        while len(ref_traj_pts) < self.ref_traj_len:
            ref_traj_pts.append(global_planed_traj[-1, :])
        return np.array(ref_traj_pts)