"""微调服务层单元测试"""

from unittest.mock import MagicMock, patch, PropertyMock
from pathlib import Path
import tempfile
import json

import pytest

from app.models.model_adapter import FineTuneDataset, FineTuneConfig


@pytest.fixture
def temp_db_path():
    """提供临时 SQLite 路径"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture(autouse=True)
def clean_global_datasets():
    """每测试前清空模块级 _datasets（避免测试间互相干扰）"""
    from app.services.fine_tuning_service import _datasets as ds_store
    ds_store.clear()
    yield


@pytest.fixture
def service(temp_db_path):
    """创建 FineTuningService 实例，使用临时目录并 mock 掉 SessionMemoryStore"""
    with patch("app.services.fine_tuning_service.SessionMemoryStore") as MockStore:
        mock_instance = MagicMock()
        MockStore.return_value = mock_instance

        from app.services.fine_tuning_service import FineTuningService

        svc = FineTuningService()
        # 使用临时目录作为 models 目录
        svc.models_dir = Path(temp_db_path)
        # 清空任务以便测试
        svc.tasks = {}
        yield svc


class TestUploadDataset:
    """upload_dataset 方法测试"""

    def test_upload_valid_data(self, service):
        data = [{"input": "hello", "output": "world"}]
        dataset = service.upload_dataset("test_ds", data)
        assert dataset.name == "test_ds"
        assert len(dataset.data) == 1

    def test_upload_multiple_items(self, service):
        data = [
            {"input": "q1", "output": "a1"},
            {"input": "q2", "output": "a2"},
            {"input": "q3", "output": "a3"},
        ]
        dataset = service.upload_dataset("multi", data)
        assert len(dataset.data) == 3

    def test_upload_empty_raises(self, service):
        with pytest.raises(ValueError, match="不能为空"):
            service.upload_dataset("empty", [])

    def test_upload_invalid_format_raises(self, service):
        with pytest.raises(ValueError, match="格式错误"):
            service.upload_dataset("bad", ["not a dict"])

    def test_upload_missing_fields_raises(self, service):
        with pytest.raises(ValueError, match="缺少"):
            service.upload_dataset("bad", [{"input": "only"}])

    def test_upload_same_name_overwrites(self, service):
        data1 = [{"input": "a", "output": "b"}]
        data2 = [{"input": "c", "output": "d"}]
        service.upload_dataset("same_name", data1)
        service.upload_dataset("same_name", data2)
        datasets = service.list_datasets()
        matching = [d for d in datasets if d["name"] == "same_name"]
        assert len(matching) == 1
        assert matching[0]["size"] == 1  # 被覆盖


class TestListDatasets:
    """list_datasets 方法测试"""

    def test_empty_when_none_uploaded(self, service):
        datasets = service.list_datasets()
        assert datasets == []

    def test_after_upload(self, service):
        service.upload_dataset("ds1", [{"input": "a", "output": "b"}])
        service.upload_dataset("ds2", [{"input": "c", "output": "d"}])
        datasets = service.list_datasets()
        assert len(datasets) == 2
        names = {d["name"] for d in datasets}
        assert names == {"ds1", "ds2"}

    def test_list_includes_size(self, service):
        data = [{"input": f"q{i}", "output": f"a{i}"} for i in range(10)]
        service.upload_dataset("ten", data)
        datasets = service.list_datasets()
        matching = [d for d in datasets if d["name"] == "ten"]
        assert matching[0]["size"] == 10

    def test_list_includes_created_at(self, service):
        service.upload_dataset("ts_ds", [{"input": "a", "output": "b"}])
        datasets = service.list_datasets()
        assert datasets[0]["created_at"] is not None
        assert len(datasets[0]["created_at"]) > 0


class TestGetDataset:
    """get_dataset 方法测试"""

    def test_get_existing(self, service):
        service.upload_dataset("my_ds", [{"input": "a", "output": "b"}])
        ds = service.get_dataset("my_ds")
        assert ds is not None
        assert ds.name == "my_ds"

    def test_get_nonexistent(self, service):
        ds = service.get_dataset("nonexistent")
        assert ds is None

    def test_get_after_overwrite(self, service):
        service.upload_dataset("ds", [{"input": "old", "output": "old"}])
        service.upload_dataset("ds", [{"input": "new", "output": "new"}])
        ds = service.get_dataset("ds")
        assert ds.data[0]["input"] == "new"


class TestTaskManagement:
    """微调任务管理测试"""

    def test_start_task_returns_task_id(self, service):
        adapter = MagicMock()
        adapter.is_fine_tunable.return_value = True
        dataset = FineTuneDataset("ds", [{"input": "a", "output": "b"}])
        config = FineTuneConfig(epochs=1)
        task_id = service.start_fine_tuning(adapter, dataset, config)
        assert task_id is not None
        assert len(task_id) > 0

    def test_start_task_stores_task(self, service):
        adapter = MagicMock()
        adapter.is_fine_tunable.return_value = True
        dataset = FineTuneDataset("ds", [{"input": "a", "output": "b"}])
        config = FineTuneConfig(epochs=1)
        task_id = service.start_fine_tuning(adapter, dataset, config)

        status = service.get_task_status(task_id)
        assert status is not None
        assert status["task_id"] == task_id

    def test_task_not_failed_on_start(self, service):
        """新建任务初始状态不应为 failed"""
        _, task = _make_task_directly(service)
        assert task.status != "failed"
        assert task.status in ("pending",)

    def test_task_contains_dataset_info(self, service):
        adapter = MagicMock()
        adapter.is_fine_tunable.return_value = True
        dataset = FineTuneDataset("my_ds", [{"input": "a", "output": "b"}])
        config = FineTuneConfig(epochs=1)
        task_id = service.start_fine_tuning(adapter, dataset, config)

        status = service.get_task_status(task_id)
        assert status["dataset_info"]["name"] == "my_ds"
        assert status["dataset_info"]["size"] == 1

    def test_task_contains_config(self, service):
        adapter = MagicMock()
        adapter.is_fine_tunable.return_value = True
        dataset = FineTuneDataset("ds", [{"input": "a", "output": "b"}])
        config = FineTuneConfig(epochs=5, batch_size=8, learning_rate=1e-4, use_lora=False)
        task_id = service.start_fine_tuning(adapter, dataset, config)

        status = service.get_task_status(task_id)
        assert status["config"]["epochs"] == 5
        assert status["config"]["batch_size"] == 8
        assert status["config"]["use_lora"] is False

    def test_non_finetunable_model_raises(self, service):
        adapter = MagicMock()
        adapter.is_fine_tunable.return_value = False
        dataset = FineTuneDataset("ds", [{"input": "a", "output": "b"}])
        config = FineTuneConfig(epochs=1)
        with pytest.raises(ValueError, match="不支持微调"):
            service.start_fine_tuning(adapter, dataset, config)

    def test_cancel_running_task(self, service):
        adapter = MagicMock()
        adapter.is_fine_tunable.return_value = True
        dataset = FineTuneDataset("ds", [{"input": "a", "output": "b"}])
        config = FineTuneConfig(epochs=1)
        task_id = service.start_fine_tuning(adapter, dataset, config)

        # 直接设置任务状态为 running
        service.tasks[task_id].status = "running"
        result = service.cancel_task(task_id)
        assert result is True

        cancelled = service.get_task_status(task_id)
        assert cancelled["status"] == "failed"

    def test_cancel_completed_task_returns_false(self, service):
        adapter = MagicMock()
        adapter.is_fine_tunable.return_value = True
        dataset = FineTuneDataset("ds", [{"input": "a", "output": "b"}])
        config = FineTuneConfig(epochs=1)
        task_id = service.start_fine_tuning(adapter, dataset, config)

        # 直接设置任务状态为 completed
        service.tasks[task_id].status = "completed"
        result = service.cancel_task(task_id)
        assert result is False

    def test_cancel_nonexistent_task(self, service):
        result = service.cancel_task("nonexistent")
        assert result is False

    def test_get_nonexistent_task_status(self, service):
        status = service.get_task_status("nonexistent")
        assert status is None

    def test_list_tasks(self, service):
        adapter = MagicMock()
        adapter.is_fine_tunable.return_value = True
        dataset = FineTuneDataset("ds", [{"input": "a", "output": "b"}])
        config = FineTuneConfig(epochs=1)

        service.start_fine_tuning(adapter, dataset, config)
        service.start_fine_tuning(adapter, dataset, config)

        tasks = service.list_tasks()
        assert len(tasks) == 2

    def test_list_tasks_filter_by_status(self, service):
        adapter = MagicMock()
        adapter.is_fine_tunable.return_value = True
        dataset = FineTuneDataset("ds", [{"input": "a", "output": "b"}])
        config = FineTuneConfig(epochs=1)

        task_id = service.start_fine_tuning(adapter, dataset, config)
        service.tasks[task_id].status = "completed"

        pending = service.list_tasks(status="pending")
        completed = service.list_tasks(status="completed")
        assert len(pending) == 0
        assert len(completed) == 1


def _make_task_directly(service, result=None):
    """直接创建任务对象（绕过线程），用于测试 task 状态"""
    from app.services.fine_tuning_service import FineTuneTask
    adapter = MagicMock()
    adapter.is_fine_tunable.return_value = True
    dataset = FineTuneDataset("ds", [{"input": "a", "output": "b"}])
    config = FineTuneConfig(epochs=1)

    import uuid
    task_id = str(uuid.uuid4())
    task = FineTuneTask(task_id, adapter, dataset, config)
    service.tasks[task_id] = task
    return task_id, task


class TestTaskSerialization:
    """FineTuneTask.to_dict 序列化测试"""

    def test_to_dict_includes_all_fields(self, service):
        adapter = MagicMock()
        adapter.is_fine_tunable.return_value = True
        dataset = FineTuneDataset("ds", [{"input": "a", "output": "b"}])
        config = FineTuneConfig(epochs=1)
        task_id = service.start_fine_tuning(adapter, dataset, config)

        task = service.tasks[task_id]
        d = task.to_dict()
        assert "task_id" in d
        assert "status" in d
        assert "progress" in d
        assert "created_at" in d
        assert "dataset_info" in d
        assert "config" in d

    def test_to_dict_result_none_by_default(self, service):
        _, task = _make_task_directly(service)
        assert task.result is None
        assert task.error is None

    def test_to_dict_includes_timestamps(self, service):
        adapter = MagicMock()
        adapter.is_fine_tunable.return_value = True
        dataset = FineTuneDataset("ds", [{"input": "a", "output": "b"}])
        config = FineTuneConfig(epochs=1)
        task_id = service.start_fine_tuning(adapter, dataset, config)

        task = service.tasks[task_id]
        task.status = "completed"
        task.started_at = "2026-01-01T00:00:00"
        task.completed_at = "2026-01-01T01:00:00"

        d = task.to_dict()
        assert d["started_at"] == "2026-01-01T00:00:00"
        assert d["completed_at"] == "2026-01-01T01:00:00"


class TestEvaluateModel:
    """evaluate_model 方法测试"""

    def test_evaluate_calls_adapter(self, service):
        adapter = MagicMock()
        adapter.evaluate.return_value = {"status": "success", "average_perplexity": 10.5}

        dataset = FineTuneDataset("eval_ds", [{"input": "a", "output": "b"}])
        result = service.evaluate_model(adapter, dataset)

        adapter.evaluate.assert_called_once_with(dataset)
        assert result["status"] == "success"
        assert result["average_perplexity"] == 10.5

    def test_evaluate_adapter_error(self, service):
        adapter = MagicMock()
        adapter.evaluate.side_effect = RuntimeError("evaluation failed")

        dataset = FineTuneDataset("bad", [{"input": "a", "output": "b"}])
        result = service.evaluate_model(adapter, dataset)

        assert result["status"] == "failed"
        assert "evaluation failed" in result["error"]


class TestGetAvailableModels:
    """get_available_models 方法测试"""

    def test_no_models_in_empty_dir(self, service):
        models = service.get_available_models()
        assert models == []

    def test_detects_peft_model(self, service, temp_db_path):
        model_dir = Path(temp_db_path) / "peft_model"
        model_dir.mkdir()
        config_path = model_dir / "adapter_config.json"
        config_path.write_text(
            json.dumps({"model_name": "gpt2", "is_peft": True, "device": "cpu"})
        )

        service.models_dir = Path(temp_db_path)
        models = service.get_available_models()
        assert len(models) >= 1
        assert any(m["name"] == "peft_model" for m in models)

    def test_detects_gpt_tiny_model(self, service, temp_db_path):
        model_dir = Path(temp_db_path) / "gpt_tiny_model"
        model_dir.mkdir()
        (model_dir / "GPT-tiny.pt").write_text("dummy weights")
        (model_dir / "dict_datas.json").write_text(json.dumps({"dummy": True}))

        service.models_dir = Path(temp_db_path)
        models = service.get_available_models()
        assert len(models) >= 1
        assert any(m["name"] == "gpt_tiny_model" for m in models)

    def test_skips_non_model_dirs(self, service, temp_db_path):
        model_dir = Path(temp_db_path) / "not_a_model"
        model_dir.mkdir()
        # 没有 adapter_config.json 也没有 GPT-tiny.pt
        (model_dir / "some_file.txt").write_text("hello")

        service.models_dir = Path(temp_db_path)
        models = service.get_available_models()
        matching = [m for m in models if m["name"] == "not_a_model"]
        assert len(matching) == 0
