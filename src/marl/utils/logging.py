"""简单的日志工具。"""

import logging
from pathlib import Path


def setup_logging(log_dir: Path) -> None:
    """配置控制台与文件日志。"""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "marl.log"

    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(log_path, mode="a", encoding="utf-8"),
    ]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


__all__ = ["setup_logging"]

