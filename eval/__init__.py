import os

# macOS: torch(임베딩)와 faiss가 각자 libomp를 로드해 멀티스레드 검색(asyncio.to_thread) 중 세그폴트 발생
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
