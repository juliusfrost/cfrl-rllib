import sys
import time
sys.path.append('.')

from driving import env_creator
import numpy as np  
env = env_creator()
env.reset()
env.render()
done = False
while not done:
    # action = env.action_space.sample()
    action = [0,0]
    a = input()
    # action[0] = +left -right, action[1] = +forward, -backward
    if 'a' in a:
        action[0] += 1
    if 'd' in a:
        action[0] -= 1
    if 'w' in a:
        action[1] += 1
    if 's' in a:
        action[1] -= 1
    if 'v' in a:
        seed = np.random.randint(1e8)
        np.random.seed(seed)
        saved_state = env.game_state.game.getGameState()
    if 'r' in a:
        np.random.seed(seed)
        env.game_state.game.setGameState(saved_state)
    _, _, done, _ = env.step(action)
    env.render()

