"""
通信自适应多智能体强化学习（MARL）框架。

该包提供层级通信、训练流水线、策略模块与环境注册的基本骨架，
便于在本地快速迭代并迁移到 HPC 集群进行大规模实验。
"""

from importlib import metadata


def get_version() -> str:
    """返回包版本；如果未安装则默认返回开发版本。"""
    try:
        return metadata.version("marl")
    except metadata.PackageNotFoundError:
        return "0.0.0-dev"


__all__ = ["get_version"]

