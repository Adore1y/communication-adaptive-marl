"""轻量环境注册实现。"""

from __future__ import annotations

from typing import Callable, Dict

ENV_REGISTRY: Dict[str, Callable[[], object]] = {}


def register_env(name: str, factory: Callable[[], object]) -> None:
    """注册环境构造器。"""
    if name in ENV_REGISTRY:
        raise ValueError(f"环境 {name} 已存在，请勿重复注册。")
    ENV_REGISTRY[name] = factory


def make_env(name: str):
    """根据名称构造环境。"""
    if name not in ENV_REGISTRY:
        raise KeyError(f"环境 {name} 未注册，请先调用 register_env。")
    return ENV_REGISTRY[name]()


# 预注册 MiniGrid 示例，按照项目计划的最小复现进行占位。
try:
    import gymnasium as gym  # type: ignore

    register_env(
        "minigrid_doorkey",
        lambda: gym.make("MiniGrid-DoorKey-5x5-v0", render_mode=None),
    )
except Exception:  # pragma: no cover - 仅在本地可用时注册
    pass


__all__ = ["ENV_REGISTRY", "register_env", "make_env"]

