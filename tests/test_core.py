from anc.core.interfaces import Configuration, Result
from typing import Any, Dict

class MockConfig(Configuration):
    def load(self, filepath: str) -> None:
        pass
    def get(self, key: str, default: Any = None) -> Any:
        return default

class MockResult(Result):
    def to_dict(self) -> Dict[str, Any]:
        return {"mock": True}

def test_config_interface():
    config = MockConfig()
    assert config.get("test", "default") == "default"

def test_result_interface():
    res = MockResult()
    assert res.to_dict()["mock"] is True
