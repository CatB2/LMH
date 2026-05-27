"""测试共享配置和 Fixtures"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

# 确保项目根目录在 sys.path 中
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# 覆盖配置环境变量（在所有应用代码 import 之前）
os.environ.setdefault("DASHSCOPE_API_KEY", "test-key")
os.environ.setdefault("FINE_TUNE_MODELS_DIR", "/tmp/test_models")


@pytest.fixture
def temp_models_dir():
    """提供临时模型目录"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)
