# Collect (frame, action, reward) transitions from the randomized-exit player with a uniform random policy.
# Ground truth (robot cell, exit cell) is recovered from the pixels themselves and validated against the real rewards.
import sys, numpy as np
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
out=sys.argv[1]; N=int(sys.argv[2]); seed=int(sys.argv[3]); player=sys.argv[4] if len(sys.argv)>4 else 'Builds/EscapeRoomRandomExit.app'
eng=EngineConfigurationChannel(); eng.set_configuration_parameters(width=256,height=256,time_scale=20,quality_level=0)
par=EnvironmentParametersChannel(); par.set_float_parameter('reset_seed',seed); par.set_float_parameter('exit_set',0)
env=UnityEnvironment(player,seed=seed,side_channels=[eng,par],no_graphics=False,timeout_wait=120,
    additional_args=['-batchmode','-screen-fullscreen','0','-screen-width','256','-screen-height','256'],log_folder=out+'-logs')
rng=np.random.default_rng(seed); O=[];A=[];R=[];T=[];S=[]   # obs, action, reward, terminal?, step index in episode
try:
    env.reset(); b=next(iter(env.behavior_specs)); prev=None; step=0
    while len(O)<N:
        d,t=env.get_steps(b)
        if len(t):
            if prev is not None: O.append(prev[0]);A.append(prev[1]);R.append(float(t.reward[0]));T.append(1);S.append(step)
            prev=None; step=0
        if len(d):
            if prev is not None: O.append(prev[0]);A.append(prev[1]);R.append(float(d.reward[0]));T.append(0);S.append(step)
            a=int(rng.integers(0,4)); prev=(d.obs[0][0].copy(),a); step+=1
            env.set_actions(b,ActionTuple(discrete=np.array([[a]],dtype=np.int32)))
        env.step()
finally: env.close()
np.savez_compressed(out,obs=np.stack(O).astype(np.float16),act=np.array(A),rew=np.array(R),term=np.array(T),step=np.array(S))
print("collected",len(O),"transitions; episodes ended:",int(np.sum(T)))
