"""
TODO: Missing module docstring
"""

import numpy as np

trial_vel = {
    0: 2,
    1: -1,
    2: -2,
    3: 1,
    4: -3,
    5: 3,
}

def compute_positions(env_width, pos_noise, 
    num_data_points, random_state, trial):
    
    half_env_width = env_width/2
    positions = np.zeros(num_data_points)
    if pos_noise == 0:
        pos = 0
    else:
        pos = random_state.randint(-pos_noise, pos_noise)
    # random between ±(1,3)
    # vel = 1. * (1. + 2. * random_state.random()) * random_state.choice([-1,1])
    # vel = 1. if trial%2 == 0 else -1
    vel = trial_vel[trial]
    # print("velocity", vel)
    positions[0] = pos        
    next_boundary_pos = lambda : np.sign(vel) * (half_env_width - random_state.random()*pos_noise)
    boundary_pos = next_boundary_pos() # init bounday with noise
    # print('next boundary', boundary_pos)
    for i in range(1,num_data_points):      
        pos += vel        
        pos_sign, pos_abs = np.sign(pos), np.abs(pos)
        boundary_pos_sign, boundary_pos_abs = np.sign(boundary_pos), np.abs(boundary_pos)                
        positions[i] = pos        
        if pos_sign==boundary_pos_sign and pos_abs >= boundary_pos_abs:
            sign = np.sign(pos)
            overstep = pos_abs - boundary_pos_abs
            pos = sign * (boundary_pos_abs - overstep) # go back by overstep
            vel = -vel # invert velocity            
            boundary_pos = next_boundary_pos() # compute next boundary with noise
            # print('next boundary', boundary_pos)                
        # print(pos)            
    return positions

def compute_positions_constant(env_width, num_data_points):
    half_env_width = env_width/2
    positions = np.zeros(num_data_points)
    # generate 100 pos noise to be consumed later
    pos = 0
    vel = 1
    for i in range(num_data_points):      
        positions[i] = pos
        pos += vel
        pos_abs = np.abs(pos)
        if pos_abs >= half_env_width:
            sign = np.sign(pos)
            overstep = half_env_width - pos_abs
            pos = sign * half_env_width - overstep # go back by overstep
            vel = -vel # invert velocity            
    return positions

if __name__ == "__main__":
    positions = compute_positions(
        env_width=100, 
        pos_noise=0, 
        num_data_points=500, 
        random_state=np.random.RandomState(0)
    )
    print(positions)