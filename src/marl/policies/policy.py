"""策略骨架。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Protocol


class BasePolicy(Protocol):
    """所有策略应实现的接口。"""

    def act(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """根据观测返回动作与可选的通信向量。"""
        ...


@dataclass
class RandomPolicy:
    """基础随机策略（占位）。"""

    action_dim: int = 4

    def act(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "action": [0.0] * self.action_dim,
            "communicate": observation.get("uncertainty", 0.0) > 0.5,
        }


class PolicyFactory:
    """策略工厂，便于按名称构造策略。"""

    _registry: Dict[str, Any] = {"random": RandomPolicy}

    @classmethod
    def register(cls, name: str, policy_cls: Any) -> None:
        if name in cls._registry:
            raise ValueError(f"策略 {name} 已存在。")
        cls._registry[name] = policy_cls

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> BasePolicy:
        if name not in cls._registry:
            raise KeyError(f"策略 {name} 未注册。")
        return cls._registry[name](**kwargs)


__all__ = ["BasePolicy", "RandomPolicy", "PolicyFactory"]

