from typing import Any

import gym


class SimulatorStateWrapper(gym.Wrapper):
    """
    An abstract simulator state gym wrapper API.
    All environments should provide their own subclassed version of this and include it in their factory method.
    """

    def get_simulator_state(self) -> Any:
        """
        :return: the simulator state at the current time step used for load_simulator_state
        """
        raise NotImplementedError

    def load_simulator_state(self, state: Any) -> bool:
        """
        :param state: should be the output of get_simulator_state
        :return: True if load was successful
        """
        raise NotImplementedError
