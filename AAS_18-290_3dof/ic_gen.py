import numpy as np
import attitude_utils as attu

class Landing_icgen(object):

    def __init__(self, downrange = (500,1500 , -70, -10), crossrange = (-500,500 , -30,30),  altitude = (1900,2100,-90,-70) , scale=None, 
                 noise_u=np.zeros(3), noise_sd=np.zeros(3), mass_uncertainty=0.0, g_uncertainty=(0.0, 0.0), adjust_apf_v0=True): 
        self.downrange = downrange
        self.crossrange = crossrange
        self.altitude = altitude
        self.noise_u = noise_u
        self.noise_sd = noise_sd
        self.mass_uncertainty = mass_uncertainty
        self.g_uncertainty = g_uncertainty
        self.nominal_mass = 2000.
        self.nominal_g = -3.7114
        self.adjust_apf_v0 = adjust_apf_v0 

        assert downrange[0] <= downrange[1]
        assert downrange[2] <= downrange[3]
        assert crossrange[0] <= crossrange[1]
        assert crossrange[2] <= crossrange[3]
        assert altitude[0] <= altitude[1]
        assert altitude[2] <= altitude[3]

        if scale is not None:
            self.downrange = tuple([x / scale for x in downrange])
            self.crossrange = tuple([x / scale for x in crossrange])
            self.altitude = tuple([x / scale for x in altitude])
 

    def show(self):
        print('Landing_icgen:')
        print('    downrange                   : ',self.downrange)
        print('    crossrange                  : ',self.crossrange)
        print('    altitude                    : ',self.altitude)
        print('    adjust_apf_v0               : ',self.adjust_apf_v0) 
    def set_ic(self , lander, dynamics):

        dynamics.noise_u = np.random.uniform(low=-self.noise_u, high=self.noise_u,size=3)
        dynamics.noise_sd = self.noise_sd

        dynamics.g[2] =  np.random.uniform(low=self.nominal_g * (1 - self.g_uncertainty[0]), 
                                           high=self.nominal_g * (1 + self.g_uncertainty[0]))

        dynamics.g[0:2] = np.random.uniform(low=-self.nominal_g * self.g_uncertainty[1],
                                            high=self.nominal_g * self.g_uncertainty[1],
                                            size=2)
                                           
        lander.init_mass = np.random.uniform(low=self.nominal_mass * (1 - self.mass_uncertainty), 
                                             high=self.nominal_mass * (1 + self.mass_uncertainty))
     
        r_downrange = np.random.uniform(low=self.downrange[0], high=self.downrange[1])
        r_crossrange = np.random.uniform(low=self.crossrange[0], high=self.crossrange[1])
        r_altitude = np.random.uniform(low=self.altitude[0], high=self.altitude[1])

        v_downrange = np.random.uniform(low=self.downrange[2], high=self.downrange[3])
        v_crossrange = np.random.uniform(low=self.crossrange[2], high=self.crossrange[3])
        v_altitude = np.random.uniform(low=self.altitude[2], high=self.altitude[3])

        lander.state['position'] = np.asarray([r_downrange,r_crossrange,r_altitude])
        lander.state['velocity'] = np.asarray([v_downrange,v_crossrange,v_altitude])
        lander.state['thrust'] = np.ones(3)*lander.min_thrust 
        lander.state['mass']   = lander.init_mass

        lander.apf_state = False
        if self.adjust_apf_v0:
            lander.apf_v0 = np.linalg.norm(lander.state['velocity'])
