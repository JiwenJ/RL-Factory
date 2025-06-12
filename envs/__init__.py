from .base import Env as BaseEnv
from .search import SearchEnv
from .reward_rollout_example import RewardRolloutEnv
from .search_vl import SearchEnvVL
__all__ = ['BaseEnv', 'SearchEnv', 'RewardRolloutEnv']

TOOL_ENV_REGISTRY = {
    'base': BaseEnv,
    'search': SearchEnv,
    'reward_rollout': RewardRolloutEnv,
    'search_vl': SearchEnvVL,


}