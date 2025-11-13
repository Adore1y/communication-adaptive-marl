"""训练流水线骨架。"""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Optional

from ..communication.hierarchical import (
    AgentCommunicator,
    GlobalCoordinator,
    RegionalRelay,
)
from ..config import TrainingConfig
from ..envs import make_env
from ..policies import PolicyFactory
from ..utils.logging import setup_logging

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """管理环境、策略与通信模块的最小训练骨架。"""

    def __init__(self, config: Optional[TrainingConfig] = None):
        self.config = config or TrainingConfig()
        setup_logging(self.config.log_dir)
        self.env = make_env(self.config.env_id)
        self.global_coord = GlobalCoordinator()
        self.regional_relay = RegionalRelay()
        self.agent_comm = AgentCommunicator()
        self.policy = PolicyFactory.create("random")

    def run(self) -> None:
        """执行占位训练循环：演示模块交互流程。"""
        logger.info("启动训练流水线：%s", asdict(self.config))
        obs, _ = self.env.reset()
        for step in range(5):  # 占位小循环
            global_signal = self.global_coord.broadcast({"step": step})
            relay_signal = self.regional_relay.route(global_signal, [obs])
            communicate, message = self.agent_comm.decide(obs, relay_signal[0])
            action_dict = self.policy.act({"uncertainty": obs.get("uncertainty", 0.0)})
            action = action_dict["action"]
            obs, reward, terminated, truncated, _ = self.env.step(action)
            logger.debug(
                "step=%d reward=%.3f communicate=%s message=%s",
                step,
                float(reward),
                communicate,
                message,
            )
            if terminated or truncated:
                obs, _ = self.env.reset()
        logger.info("占位训练完成，可替换为实际RL训练逻辑。")


__all__ = ["TrainingPipeline"]


if __name__ == "__main__":
    TrainingPipeline().run()

