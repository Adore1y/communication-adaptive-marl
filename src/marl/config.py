"""训练与通信配置的数据结构。"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class CommunicationBudget:
    """通信预算与门控相关超参。"""

    budget_per_10k: int = 5
    confidence_threshold: float = 0.7
    relay_group_size: int = 8


@dataclass
class TrainingConfig:
    """整体训练配置。"""

    algo: str = "MAPPO"
    env_id: str = "minigrid_doorkey"
    total_timesteps: int = 1_000_000
    num_envs: int = 16
    rollout_length: int = 256
    seed: int = 0
    log_dir: Path = Path("logs")
    checkpoint_dir: Path = Path("checkpoints")
    eval_interval: int = 50_000
    use_memory: bool = True
    use_reflection: bool = True
    use_tools: bool = True
    communication: CommunicationBudget = field(default_factory=CommunicationBudget)
    extras: Optional[List[str]] = None


DEFAULT_CONFIG = TrainingConfig()


__all__ = ["CommunicationBudget", "TrainingConfig", "DEFAULT_CONFIG"]

