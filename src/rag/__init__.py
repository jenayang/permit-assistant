"""RAG(검색-증강생성) 파이프라인 - retriever/reranker/ingestion/law_chunker/
parent_store/regulation을 한곳에 묶는다. 원래 src/ 최상위에 흩어져 있던 파일들인데,
서로를 실제로 호출하며 하나의 검색 파이프라인을 이루므로 이 폴더로 옮겼다.

여기서 재노출은 하지 않는다 - 호출부가 전부 이미 하위 모듈 경로를 직접 쓰고
있어서(`from src.rag.retriever import search_with_score`), 이름을 옮겨적는
껍데기만 늘어난다. `from src.rag import parent_store` 같은 서브모듈 import는
파이썬이 알아서 처리하므로 재노출이 필요 없다.
"""
