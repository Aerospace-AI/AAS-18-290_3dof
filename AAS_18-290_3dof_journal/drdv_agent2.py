import numpy as np
import env_utils as envu
import attitude_utils as attu
from time import time

class DRDV_agent(object):

    def __init__(self, env):
        self.env = env
        self.state = {}
        self.prev_state = {}
        self.mass = 2000.

    def get_thrust(self,state):
        rg = state[0:3]
        vg = state[3:6] - np.asarray([0.0,0.0,-1.0])
        gamma = 0.0
        #p = [gamma + np.linalg.norm(self.env.dynamics.g)**2/2  ,  0., -2. * np.dot(vg,vg)  , -12. * np.dot(vg,rg) , -18. * np.dot(rg , rg)]
        g = np.asarray([0.0,0.0,-3.7114])
        p = [gamma + np.linalg.norm(g)**2/2  ,  0., -2. * np.dot(vg,vg)  , -12. * np.dot(vg,rg) , -18. * np.dot(rg , rg)]

        #print(rg, vg, p)
        p_roots = np.roots(p)
        for i in range(len(p_roots)):
            if np.abs(np.imag(p_roots[i])) < 0.0001:
                if p_roots[i] > 0:
                    t_go = np.real(p_roots[i])
        #print(t_go)            
        if t_go > 0:
            a_c = -6. * rg/t_go**2 - 4. * vg /t_go - self.env.dynamics.g
        else:
            a_c = np.zeros(3) 

        #thrust = a_c * self.env.lander.state['mass']
        thrust = a_c * self.mass

        thrust = envu.limit_thrust(thrust, self.env.lander.min_thrust, self.env.lander.max_thrust)

        return thrust 

    def sample(self,state):
        return self.get_thrust(state)

    def get_state_dynamics(self):
        state = np.hstack((self.state['position'], self.state['velocity'], self.state['mass']))
        return state
 
    def test(self,render=True):
        step = 0
        state = self.env.reset()
        done = False
        t0 = time()
        while not done:
            action = self.get_thrust(state)
            prev_action = action.copy()
            state, reward, done, _  = self.env.step(action)
            step += 1
        if render:
            envu.render_traj(self.env.lander.trajectory)
        print(step, np.linalg.norm(state[0:3]), np.linalg.norm(state[3:6]))

    def test_batch(self,render=False,n=10):
        positions = []
        velocities = []
        trajectories = []
        for i in range(n):
            self.test(render)
            r_f = self.env.lander.trajectory['position'][-1]
            v_f = self.env.lander.trajectory['velocity'][-1]
            #print(r_f, v_f)
            positions.append(r_f)
            velocities.append(v_f)
            trajectories.append(self.env.lander.trajectory.copy())
        print(np.mean(positions), np.std(positions), np.max(positions))
        print(np.mean(velocities), np.std(velocities), np.max(velocities))

        return positions, velocities, trajectories
