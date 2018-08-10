import env_utils as envu
import numpy as np
from time import time

class Dynamics_model(object):

    """
        The dynamics model take a lander model object (and later an obstacle object) and modifies  
        the state of the lander.

        The lander object instantiates an engine model, that maps body frame thrust and torque to
        the inertial frame.  Note that each lander can have its own intertial frame which can be 
        centered on the lander's target. 

        Currentlly this model does not model environmental dynamics, will be added later
 
        The lander model maintains a state vector: 
            position                                [0:3]
            velocity                                [3:6]
            mass                                    [7]     
                 

    """

    def __init__(self, h=5e-2, noise_u=np.zeros(3),noise_sd=np.zeros(3) ):
        self.h = h
        self.Isp = 210.0
        self.g_o = 9.81
        self.g = np.asarray([0.0,0.0,-3.7114])
        self.noise_sd = noise_sd 
        self.noise_u =  noise_u
        print('3-dof dynamics model')

    def next(self,t,thrust_cmd,lander):
        t0 = time()
        #lander.prev_state = lander.state.copy()
        pos = lander.state['position']
        vel = lander.state['velocity']

        thrust = envu.limit_thrust(thrust_cmd, lander.min_thrust, lander.max_thrust)
        noise = (self.noise_u + self.noise_sd * np.random.normal(size=3)) /  lander.state['mass']
        #
        # Use 4th order Runge Kutta to integrate equations of motion
        #

        x = lander.get_state_dynamics()
        ode = lambda t,x : self.eqom(t, x, thrust, noise)
        x_next = envu.rk4(t, x, ode, self.h )

        lander.state['position'] = x_next[0:3]
        lander.state['velocity'] = x_next[3:6]
        lander.state['mass']     = x_next[6]
        lander.state['thrust'] = thrust 
        return x_next

    
     
           
    def eqom(self,t, x, thrust, noise):

        r = x[0:3]
        v = x[3:6]
        m = x[6]
        xdot = np.zeros(7)
        xdot[0:3] = v
        xdot[3:6] = thrust / m + self.g  + noise
        xdot[6] = -np.linalg.norm(thrust) / (self.Isp * self.g_o) 

        return xdot
 
       
        
        
