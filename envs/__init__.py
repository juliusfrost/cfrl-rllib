import envs.minigrid
import envs.driving


def register(**kwargs):
    envs.minigrid.register(**kwargs)
    envs.driving.register(**kwargs)
