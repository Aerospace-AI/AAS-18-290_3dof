import numpy as np
import env_utils as envu

class Reward(object):

    """
        Minimizes Velocity Field Tracking Error

    """

    def __init__(self, reward_scale=1.0, ts_coeff=-0.0,  fuel_coeff=-0.05, term_fuel_coeff=-0.0,
                 landing_rlimit=5.0, landing_vlimit=2.0,
                 tracking_coeff=-0.01,  tracking_bias=0.0, scaler=None):

        self.reward_scale =         reward_scale
        self.ts_coeff =             ts_coeff
        self.fuel_coeff =           fuel_coeff

        self.landing_rlimit =       landing_rlimit
        self.landing_vlimit =       landing_vlimit

        self.tracking_coeff =       tracking_coeff

        self.tracking_bias  =  tracking_bias

        self.scaler = scaler

        print('dvec vc 3')

    def get(self, lander,  action, done, steps, shape_constraint, glideslope_constraint):
        pos         =  lander.state['position']
        vel         =  lander.state['velocity']

        prev_pos    =  lander.prev_state['position']
        prev_vel    =  lander.prev_state['velocity']

        state = np.hstack((pos,vel))
        prev_state = np.hstack((prev_pos,prev_vel))

        r_gs = glideslope_constraint.get_reward()

        r_sc, sc_margin = shape_constraint.get_reward(lander.state)

        error, t_go = lander.track_func(pos,vel)
        r_tracking  = self.tracking_bias + self.tracking_coeff * np.linalg.norm(error) 

        r_fuel = self.fuel_coeff * np.linalg.norm(action) / lander.max_thrust

        r_landing = 0.
        landing_margin = 0.
        gs_penalty = 0.0
        sc_penalty = 0.0
        if done:

            gs_penalty = glideslope_constraint.get_term_reward()

            sc_penalty = shape_constraint.get_term_reward(lander.state)

            landing_margin = np.maximum(np.linalg.norm(pos) -  self.landing_rlimit , np.linalg.norm(vel) -  self.landing_vlimit)

        reward_info = {}

        reward_info['fuel'] = r_fuel

        reward = (sc_penalty + gs_penalty + r_gs + r_sc +  r_landing + r_tracking +  r_fuel + self.ts_coeff) * self.reward_scale

        lander.trajectory['reward'].append(reward)
        lander.trajectory['glideslope'].append(glideslope_constraint.get())
        lander.trajectory['glideslope_reward'].append(r_gs)
        lander.trajectory['glideslope_penalty'].append(gs_penalty)
        lander.trajectory['sc_penalty'].append(sc_penalty)
        lander.trajectory['sc_margin'].append(sc_margin)
        lander.trajectory['sc_reward'].append(r_sc)
        lander.trajectory['landing_reward'].append(0.0)
        lander.trajectory['tracking_reward'].append(r_tracking)
        lander.trajectory['landing_margin'].append(landing_margin)
        lander.trajectory['range_reward'].append(0.0)
        lander.trajectory['fuel_reward'].append(r_fuel)
        return reward, reward_info


