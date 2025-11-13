"""针对训练流水线的最小单元测试。"""

from typing import Any, Dict, Tuple

from marl.config import TrainingConfig
from marl.envs import register_env
from marl.training.pipeline import TrainingPipeline


class _DummyEnv:
    def reset(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        return {"uncertainty": 0.6}, {}

    def step(
        self, action
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        return {"uncertainty": 0.4}, 1.0, True, False, {}


def test_pipeline_runs_minimal_steps(tmp_path):
    try:
        register_env("dummy_env", _DummyEnv)
    except ValueError:
        pass
    config = TrainingConfig(
        env_id="dummy_env", total_timesteps=10, log_dir=tmp_path / "logs"
    )
    pipeline = TrainingPipeline(config)
    pipeline.run()

