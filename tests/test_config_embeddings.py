import ast
from pathlib import Path

from src.embeddings import pick_device


def test_config_has_single_definitions():
    tree = ast.parse(Path("src/config.py").read_text())
    funcs = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
    llm_assigns = [n for n in tree.body if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "LLM" for t in n.targets)]
    assert funcs.count("load_faiss_index") == 1
    assert len(llm_assigns) == 1


def test_pick_device_env_override(monkeypatch):
    monkeypatch.setenv("EMBED_DEVICE", "cpu")
    assert pick_device() == "cpu"


def test_pick_device_prefers_gpu(monkeypatch):
    import torch
    monkeypatch.delenv("EMBED_DEVICE", raising=False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert pick_device() == "mps"


def test_all_embedding_construction_goes_through_factory():
    for f in ["src/config.py", "src/rag_updater.py"]:
        assert "HuggingFaceEmbeddings(" not in Path(f).read_text(), f
