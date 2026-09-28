import ast
from pathlib import Path

from src.retrieval.store import pick_device


def test_config_has_single_definitions():
    tree = ast.parse(Path("src/config.py").read_text())
    llm_assigns = [n for n in tree.body if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "LLM" for t in n.targets)]
    assert len(llm_assigns) == 1
    store = ast.parse(Path("src/retrieval/store.py").read_text())
    assert [n.name for n in store.body if isinstance(n, ast.FunctionDef)].count("load_faiss_index") == 1


def test_pick_device_env_override(monkeypatch):
    monkeypatch.setenv("EMBED_DEVICE", "mps")
    assert pick_device() == "mps"


def test_pick_device_defaults_to_cpu_for_service(monkeypatch):
    """서비스 배포 대상이 CPU 서버 — 가속기는 EMBED_DEVICE로 명시할 때만."""
    import torch
    monkeypatch.delenv("EMBED_DEVICE", raising=False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert pick_device() == "cpu"


def test_all_embedding_construction_goes_through_factory():
    for f in ["src/config.py", "src/rag_updater.py", "src/retrieval/retriever.py"]:
        assert "HuggingFaceEmbeddings(" not in Path(f).read_text(), f
