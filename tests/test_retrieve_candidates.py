import pytest
from langchain_core.documents import Document

from src.retrieval import retriever as tools  # 검색 단계 전역(체인·검색 함수)이 있는 모듈을 패치
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

    for name, val in dict(region_chain=Chain("부산광역시 중구"),
                          query_gen_chain=Chain("부산 중구 맛집, 부산 중구 노포"),
                          resolve_admin_region=resolve, get_coordinates=coords,
                          calculate_haversine_distance=lambda *a: 1.0,
                          _search_docs=search, _filter_candidates=filt).items():
        monkeypatch.setattr(tools, name, val)
    return calls


async def test_multiquery_search_and_name_dedupe(fake_tools):
    r = await tools.retrieve_candidates("맛집", "부산 중구", "식당", anchor="부산 중구")
    assert r["target_region"] == "부산광역시 중구"
    assert r["queries"] == ["부산 중구 맛집", "부산 중구 노포"]
    assert fake_tools["search"] == [("부산 중구 맛집", 50), ("부산 중구 노포", 50)]
    assert [d.metadata["장소명"] for d in r["candidates"]] == ["A", "B"]
    assert r["fallback_used"] is False


async def test_region_chain_failure_falls_back_to_maps(fake_tools, monkeypatch):
    monkeypatch.setattr(tools, "region_chain", Chain(RuntimeError("llm down")))
    r = await tools.retrieve_candidates("맛집", "부산", "식당", anchor="부산")
    assert r["target_region"] == "부산광역시"


def test_doc_to_cand():
    assert legacy_adapter.doc_to_cand(doc("A", "부산광역시 중구")) == {
        "name": "A", "region": "부산광역시 중구", "category": "식당"}


async def test_adapter_delegates_to_src(fake_tools):
    r = await legacy_adapter.legacy_retrieve("맛집", "부산 중구", "식당")
    assert [d.metadata["장소명"] for d in r["candidates"]] == ["A", "B"]


async def test_primary_path_caps_at_five(fake_tools, monkeypatch):
    async def many(q, k=20):
        return [doc(f"P{i}", "부산광역시 중구") for i in range(8)]
    monkeypatch.setattr(tools, "_search_docs", many)
    r = await tools.retrieve_candidates("맛집", "부산 중구", "식당", anchor="부산 중구")
    assert len(r["candidates"]) == 5


async def test_dedupe_keeps_same_name_in_different_regions(fake_tools, monkeypatch):
    async def same_name(q, k=20):
        return [doc("원조국밥", "부산광역시 중구"), doc("원조국밥", "대구광역시 중구"), doc("원조국밥", "부산광역시 중구")]
    monkeypatch.setattr(tools, "_search_docs", same_name)
    r = await tools.retrieve_candidates("맛집", "부산 중구", "식당", anchor="부산 중구")
    assert [(d.metadata["장소명"], d.metadata["지역"]) for d in r["candidates"]] == [
        ("원조국밥", "부산광역시 중구"), ("원조국밥", "대구광역시 중구")]
