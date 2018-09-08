import numpy as np
import env_utils as envu

class Reward(object):

    """
        Minimizes Velocity Field Tracking Error

    """

    def __init__(self, reward_scale=1.0, ts_coeff=-0.0,  fuel_coeff=-0.05, term_fuel_coeff=-0.0,
                 landing_coeff=0.0 , landing_rlimit=5.0, landing_vlimit=2.0,
                 landing_pos_coeff=0.0, landing_vel_coeff=0.0, landing_gs_coeff=0.0, max_gs=10.0, 
                 tracking_coeff=-0.01,  tracking_bias=0.0, gamma=1.0, scaler=None):

        self.reward_scale =         reward_scale
        self.ts_coeff =             ts_coeff
        self.fuel_coeff =           fuel_coeff
        self.term_fuel_coeff =      term_fuel_coeff

        self.landing_coeff =        landing_coeff
        self.landing_rlimit =       landing_rlimit
        self.landing_vlimit =       landing_vlimit
        self.max_gs =               max_gs
        self.tracking_coeff =       tracking_coeff
        self.landing_pos_coeff =    landing_pos_coeff
        self.landing_vel_coeff =    landing_vel_coeff
        self.landing_gs_coeff =     landing_gs_coeff

        self.tracking_bias  =  tracking_bias

        self.terminal_phase = False 
        self.gamma = gamma

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

        if pos[2] < lander.apf_atarg:
            #if not self.terminal_phase:
            #    print('Switched')
            self.terminal_phase = True

        if self.terminal_phase:
            error = self.track_func_term(pos,vel)
        else:
            error, t_go = lander.track_func(pos,vel)

        if self.terminal_phase:
            r_tracking = 0.0
        else:
            r_tracking  = self.tracking_bias + self.tracking_coeff * np.linalg.norm(error) 

        r_fuel = self.fuel_coeff * np.linalg.norm(action) / lander.max_thrust

        r_landing = 0.
        landing_margin = 0.
        gs_penalty = 0.0
        sc_penalty = 0.0

        if done:
            self.terminal_phase = False
            gs_penalty = glideslope_constraint.get_term_reward()

            sc_penalty = shape_constraint.get_term_reward(lander.state)

            if np.linalg.norm(pos) < self.landing_rlimit and np.linalg.norm(vel) < self.landing_vlimit and pos[2] < 0.0: 
                r_landing = self.landing_gs_coeff * np.clip(glideslope_constraint.get(), 0, self.max_gs)
                r_landing += self.landing_pos_coeff * self.quad_reward2(pos,self.landing_rlimit) + self.landing_vel_coeff / 2 * self.quad_reward2(vel,self.landing_vlimit)
                                  
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
        lander.trajectory['landing_reward'].append(r_landing)
        lander.trajectory['tracking_reward'].append(r_tracking)
        lander.trajectory['landing_margin'].append(landing_margin)
        lander.trajectory['range_reward'].append(0.0)
        lander.trajectory['fuel_reward'].append(r_fuel)
        return reward, reward_info

    def quad_reward1(self,pos, vel, p_sigma, v_sigma):
        pos = np.linalg.norm(pos)
        vel = np.linalg.norm(vel)
        reward = np.exp(-pos**2/p_sigma**2 -vel**2/v_sigma**2)
        return reward

    def quad_reward2(self,val,sigma):
        val = np.linalg.norm(val)
        reward = 1 + np.maximum( -1, -(val/sigma)**2)
        return reward

    def pot_func_r(self,pos,vel,t_go):
        pos_dvec = -pos / np.linalg.norm(pos)
        vel_dvec = vel / np.linalg.norm(vel)
        pdotv = np.dot(pos_dvec,vel_dvec)
        pdotv = np.clip(pdotv,-1,1)
        t_go = np.maximum(1,t_go)
        pot = -np.arccos(pdotv) / t_go
        assert pot <=  0

        return pot

    def pot_func_v(self,pos,vel,t_go):
        mv = np.linalg.norm(vel)
        if np.linalg.norm(pos) < 100:
            pot = -mv / t_go
        else:
            pot = 0.0
        return pot

    def track_func_term(self,pos,vel):
        rg1 = 1.0*np.asarray([0,0,pos[2]])
        vg1 = vel - np.asarray([0.,0.,-1])
        mag_pot =  1.0 
        dir_pot = -rg1 / np.linalg.norm(rg1) 
        pot = mag_pot * dir_pot
        error = vel - pot
        return error 
