from typing import Dict, List

import gym
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import TensorType, try_import_torch
from ray.rllib.utils.typing import ModelConfigDict
from ray.tune.utils import merge_dicts

_, nn = try_import_torch()

DEFAULT_CONFIG = {
    'conv_filters': [
        # out_channels, kernel_size, stride, padding
        [32, [3, 3], 1, 2],
        [64, [3, 3], 3, 0],
        [128, [3, 3], 1, 2],
        [256, [3, 3], 3, 0],
    ],
    'state_embed_size': 256,
}


class MinigridConvTorchModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space: gym.spaces.Space, action_space: gym.spaces.Space, num_outputs: int,
                 model_config: ModelConfigDict, name: str):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        self.model_config = merge_dicts(DEFAULT_CONFIG, model_config['custom_model_config'])
        self.conv_filters = self.model_config['conv_filters']
        conv_list = []
        in_channels = 3
        for conv_params in self.conv_filters:
            out_channels, kernel_size, stride, padding = conv_params
            conv_list.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
            in_channels = out_channels
        self.conv_net = nn.Sequential(*conv_list)

        state_embed_size = self.model_config['state_embed_size']
        self._value_head = nn.Sequential(
            nn.Linear(state_embed_size, 1),
        )
        self._policy_head = nn.Sequential(
            nn.Linear(state_embed_size, num_outputs),
        )

    def forward(self, input_dict: Dict[str, TensorType], state: List[TensorType], seq_lens: TensorType) -> (
            TensorType, List[TensorType]):
        device = next(self.parameters()).device
        obs = input_dict['obs'].to(device).float().permute(0, 3, 1, 2)
        batch_size = obs.size(0)
        self._logits = self.conv_net(obs).reshape(batch_size, -1)
        outputs = self._policy_head(self._logits)
        return outputs, self.get_initial_state()

    def value_function(self) -> TensorType:
        values = self._value_head(self._logits)
        return values.squeeze(1)

    def import_from_h5(self, h5_file: str) -> None:
        pass


def register():
    ModelCatalog.register_custom_model('minigrid_conv_torch_model', MinigridConvTorchModel)
