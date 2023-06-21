import env.aquarium as aq
import random
import numpy as np

# This is just a very easy way to test environment function, just execute the code and random actions will be performed

env = aq.Aquarium(render_mode='human')

env.reset(seed=42)
truncated = []


for agent in env.agent_iter(env.max_timesteps):
    if agent not in truncated:
        observation, reward, termination, truncation, info = env.last()
    if termination or truncation:
        truncated.append(agent)
    # print(reward, agent, termination,
    #       observation["observation"]["surrounding"])

    if agent in truncated:
        action = None
    else:
        # this is where you would insert your policy
        # action = env.action_space(agent).sample()
        action = np.random.randint(0, 6)
    # print(action)
    env.step(action)
env.close()
