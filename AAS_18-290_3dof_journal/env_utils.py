import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def limit_thrust(thrust_cmd, min_thrust, max_thrust):
    eps = 1e-8
    thrust_mag  = np.linalg.norm(thrust_cmd)
    thrust_dvec = thrust_cmd / (thrust_mag + eps)
    thrust_mag = np.clip(thrust_mag, min_thrust, max_thrust)
    thrust = thrust_mag * thrust_dvec
    return thrust

def scale_thrust(thrust_cmd, min_thrust, max_thrust):
    thrust_mag  = np.linalg.norm(thrust_cmd)
    thrust_dvec = thrust_cmd / thrust_mag
    thrust_mag = thrust_mag * max_thrust
    thrust = thrust_mag * thrust_dvec
    return thrust

def reverse_thrust(thrust_cmd, min_thrust, max_thrust):
    thrust_mag  = np.linalg.norm(thrust_cmd)
    thrust_dvec = thrust_cmd / thrust_mag
    thrust_mag = thrust_mag / max_thrust
    thrust = thrust_mag * thrust_dvec
    return thrust


def get_glideslope(pos,vel):

    """
        Only impose constraint if: 
            above the ground
            magnitude of velocity > 1m/s

        2nd condition is to avoid noise violating the constraint when lander is
        close to making a landing

    """


    if len(vel.shape) == 1:
        dz = vel[2]
        dy = vel[1]
        dx = vel[0]
    else:
        dz = vel[:,2]
        dy = vel[:,1]
        dx = vel[:,0]

    gs = np.abs(dz) / np.sqrt(dx**2+dy**2)

    if pos[2] > 0.1 :#and np.linalg.norm(vel) > 1.0: 
        return gs
    else:
        return 100.0

def get_vc(r_tm, v_tm):
   vc = -r_tm.dot(v_tm)/np.linalg.norm(r_tm)
   return vc
 
def get_dlos(r_tm, v_tm ):
    vc = get_vc(r_tm, v_tm) 
    #dlos       =  r_tm * vc / np.linalg.norm(r_tm) ** 2
    r = np.linalg.norm(r_tm)
    if vc > 0.01:
        dlos = v_tm / r + r_tm * vc / r**2
    else:  
        dlos = np.zeros(3)
    return dlos

def get_tgo( rg, vg, g):
        gamma = 0.0
        p = [gamma + np.linalg.norm(g)**2/2  ,  0., -2. * np.dot(vg,vg)  , -12. * np.dot(vg,rg) , -18. * np.dot(rg , rg)]
        #print(rg, vg, p)
        p_roots = np.roots(p)
        for i in range(len(p_roots)):
            if np.abs(np.imag(p_roots[i])) < 0.0001:
                if p_roots[i] > 0:
                    t_go = np.real(p_roots[i])
        #print(t_go)
        if t_go < 0.:
            t_go = 0.

        return t_go 
 
def print_vector(s,v,f):
    v = 1.0 * v
    if isinstance(v,float): 
         v = [v]
    s1 = ''.join(f.format(v) for k,v in enumerate(v))
    s1 = s + s1
    return s1


def rk4(t, x, xdot, h ):

    """
        t  :  time
        x  :  initial state
        xdot: a function xdot=f(t,x, ...)
        h  : step size

    """

    k1 = h * xdot(t,x)
    k2 = h * xdot(t+h/2 , x + k1/2)
    k3 = h * xdot(t+h/2,  x + k2/2)
    k4 = h * xdot(t+h   ,  x +  k3)

    x = x + (k1 + 2*k2 + 2*k3 + k4) / 6

    return x

def get_zem_old(r_tm, v_tm):
    vc = get_vc(r_tm, v_tm)
    range = np.linalg.norm(r_tm)
    if vc > 0.1:
        t_go = get_tgo(r_tm, v_tm, 4.0) 
        zem = t_go * v_tm + r_tm
        zem = r_tm - t_go * v_tm
        #print('zem: ',zem, r_tm, t_go, v_tm)
    else:
        zem = np.zeros(3)
    #zem = np.clip(zem,-2000,2000) 
    return zem

def get_zem(r_tm, v_tm):
    vc = get_vc(r_tm, v_tm)
    range = np.linalg.norm(r_tm)
    if True: #vc > 0.1:
        t_go = get_tgo(r_tm, v_tm, 4.0)
        zem = t_go * v_tm + r_tm
        zem = r_tm - t_go * v_tm
        #print('zem: ',zem, r_tm, t_go, v_tm)
    else:
        zem = np.zeros(3)
    #zem = np.clip(zem,-2000,2000)
    return zem

def render_traj(traj, vf=None, scaler=None):

    fig1 = plt.figure(1,figsize=plt.figaspect(0.5))
    fig1.clear()
    plt.figure(fig1.number)
    fig1.set_size_inches(8, 8, forward=True)
    gridspec.GridSpec(3,2)
    t = np.asarray(traj['t'])
    pos = np.asarray(traj['position'])
    vel = np.asarray(traj['velocity'])
    norm_pos = np.linalg.norm(pos,axis=1)
    norm_vel = np.linalg.norm(vel,axis=1)

    x = pos[:,0]
    y = pos[:,1]
    z = pos[:,2]
    plt.subplot2grid( (4,2) , (0,0) )
    plt.plot(t,x,'r',label='X')
    plt.plot(t,y,'b',label='Y')
    plt.plot(t,z,'g',label='Z')
    plt.plot(t,norm_pos,'k',label='N')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
            ncol=5, mode="expand", borderaxespad=0.)
    plt.gca().set_ylabel('Position')
    plt.gca().set_xlabel("Time")
    plt.grid(True)

    vc = np.asarray(traj['vc'])
    vc = np.asarray(traj['glideslope']) 
    t1 = t[0:-1]
    plt.subplot2grid( (4,2) , (0,1) )
    colors = ['r']
    #print('debug: ',attitude.shape[1],len(colors))
    plt.plot(t1,vc,colors[0],label='vc' )
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
            ncol=5, mode="expand", borderaxespad=0.)
    plt.gca().set_ylabel('Glideslope')
    plt.gca().set_xlabel("Time")
    plt.grid(True)

    zem = np.asarray(traj['zem'])
    x = pos[:,0]
    y = pos[:,1]
    z = pos[:,2]
    xy = np.sqrt(y**2 + x**2)
    #xy = xy[::-1]
    plt.subplot2grid( (4,2) , (1,0) ) 
    if vf is not None and scaler is not None:
        state = np.hstack((pos,vel))
        values = vf.predict(scaler.apply(state))
        plt.plot(t,values,'r',label='V')
    else:
        plt.plot(xy,z,'r',label='Z')

    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
            ncol=5, mode="expand", borderaxespad=0.)
    plt.gca().set_xlabel("XY")
    plt.gca().set_ylabel('Z')
    plt.xlim(1500,0)
    plt.ylim(0,2000)
    plt.grid(True)

    r = np.asarray(traj['reward'])
    cr = np.cumsum(r)
    plt.subplot2grid( (4,2) , (1,1) )
    #print(r.shape, t1.shape)
    t1 = t[0:r.shape[0]]
    plt.plot(t1,r,'b',label='Reward')
    plt.plot(t1,cr,'g',label='Cum Reward')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
            ncol=5, mode="expand", borderaxespad=0.)
    plt.gca().set_xlabel("Time")
    plt.gca().set_ylabel('Reward')
    plt.grid(True)

    x = vel[:,0]
    y = vel[:,1]
    z = vel[:,2]

    plt.subplot2grid( (4,2) , (2,0))
    plt.plot(t,x,'r',label='X')
    plt.plot(t,y,'b',label='Y')
    plt.plot(t,z,'g',label='Z')
    plt.plot(t,norm_vel,'k',label='N')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
            ncol=5, mode="expand", borderaxespad=0.)
    plt.gca().set_ylabel('Velocity')
    plt.gca().set_xlabel("Time")
    plt.grid(True)


    thrust = np.asarray(traj['thrust'])
    x = thrust[:,0]
    y = thrust[:,1]
    z = thrust[:,2]
    plt.subplot2grid( (4,2) , (2,1) )
    plt.plot(t,x,'r',label='X')
    plt.plot(t,y,'b',label='Y')
    plt.plot(t,z,'g',label='Z')

    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
            ncol=5, mode="expand", borderaxespad=0.)
    plt.gca().set_ylabel('Thrust')
    plt.gca().set_xlabel("Time")
    plt.grid(True)

    t_go = np.asarray(traj['t_go'])
    plt.subplot2grid( (4,2) , (3,0) )
    plt.plot(t,t_go,'r',label='X')

    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
            ncol=5, mode="expand", borderaxespad=0.)
    plt.gca().set_ylabel('time-to-go')
    plt.gca().set_xlabel("Time")
    plt.grid(True)

    m = np.asarray(traj['fuel'])
    plt.subplot2grid( (4,2) , (3,1) )
    plt.plot(t,m,'r',label='X')

    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
            ncol=5, mode="expand", borderaxespad=0.)
    plt.gca().set_ylabel('Fuel')
    plt.gca().set_xlabel("Time")
    plt.grid(True)

    


    plt.tight_layout(h_pad=3.0)
    fig1.canvas.draw()

