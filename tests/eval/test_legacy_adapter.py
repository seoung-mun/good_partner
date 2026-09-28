import types
import pytest
from langchain_core.documents import Document

from eval import legacy_adapter


def doc(name, region, cat="식당"):
    return Document(page_content="x", metadata={"장소명": name, "지역": region, "카테고리": cat})


class Chain:
    def __init__(self, out):
        self.out = out

    async def ainvoke(self, _):
        if isinstance(self.out, Exception):
            raise self.out
        return self.out


@pytest.fixture
def fake_tools(monkeypatch):
    calls = {"search": []}

    async def search(q, k=20):
        calls["search"].append((q, k))
        return [doc("A", "부산광역시 중구"), doc("A", "부산광역시 중구"), doc("B", "부산광역시 중구")]

    async def filt(docs, target_region, exclude, category_filter):
        return list(docs)

    async def coords(_):
        return (35.1, 129.0)

    async def resolve(_a, dest):
        return "부산광역시"

    T = types.SimpleNamespace(
        region_chain=Chain("부산광역시 중구"), query_gen_chain=Chain("부산 중구 맛집, 부산 중구 노포"),
        resolve_admin_region=resolve, get_coordinates=coords,
        calculate_haversine_distance=lambda *a: 1.0, _search_docs=search, _filter_candidates=filt)
    monkeypatch.setattr(legacy_adapter, "_tools", lambda: T)
    return calls


async def test_multiquery_search_and_name_dedupe(fake_tools):
    r = await legacy_adapter.legacy_retrieve("맛집", "부산 중구", "식당")
    assert r["target_region"] == "부산광역시 중구"
    assert r["queries"] == ["부산 중구 맛집", "부산 중구 노포"]
    assert fake_tools["search"] == [("부산 중구 맛집", 50), ("부산 중구 노포", 50)]
    assert [d.metadata["장소명"] for d in r["candidates"]] == ["A", "B"]
    assert r["fallback_used"] is False


async def test_region_chain_failure_falls_back_to_maps(fake_tools, monkeypatch):
    T = legacy_adapter._tools()
    T.region_chain = Chain(RuntimeError("llm down"))
    r = await legacy_adapter.legacy_retrieve("맛집", "부산", "식당")
    assert r["target_region"] == "부산광역시"


def test_doc_to_cand():
    assert legacy_adapter.doc_to_cand(doc("A", "부산광역시 중구")) == {
        "name": "A", "region": "부산광역시 중구", "category": "식당"}
