"""
持久化 RAG 知识库 (ChromaDB)

基于 LlamaIndex + ChromaDB + 火山引擎 embedding(doubao-embedding-vision)，
在磁盘上持久化存储两个知识库:
  - inspection_reports: 巡检报告 (来自 config/inspector_reports/)
  - work_records: 工作记录 (来自 config/work_records/)

ChromaDB 数据存储在 config/ChromaDB/ 目录，重启不丢失。

启动时自动扫描两个目录，通过 file_fingerprint (文件名+大小+修改时间的 MD5) 去重，
只注入新文件或已变更的文件。运行时 Agent 生成的报告/记录也会同时写入磁盘和注入向量库。

使用方法:
    from utils.rag import RAGKnowledgeBase

    rag = RAGKnowledgeBase()

    # [启动时] 预加载所有文件
    await rag.init_from_disk()

    # [运行时] 写入巡检报告
    await rag.ingest_inspection(content="...", metadata={"date": "2026-05-16", "time": "14:30"})

    # [运行时] 写入工作记录
    await rag.ingest_work_record(content="...", metadata={"title": "节点资源耗尽"})

    # [运行时] 检索巡检报告（按日期过滤）
    results = await rag.retrieve_inspection(query="node3 内存", date_from="2026-05-15")

    # [运行时] 检索工作记录（按标题关键词过滤）
    results = await rag.retrieve_work_record(query="Pod被驱逐", title_keyword="资源耗尽")
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable

from llama_index.core import VectorStoreIndex, Document
from llama_index.core.node_parser import SentenceSplitter

from utils.llm import get_embedding_config
from utils.write_guard import enter_write, exit_write
from config import VECTOR_DB_DIR, INSPECTOR_REPORTS_DIR, WORK_RECORDS_DIR
from logger import get_rag_logger

_rag_log = get_rag_logger()


def _splitter_from_config(kb_name: str) -> SentenceSplitter:
    config = get_embedding_config(kb_name)
    ingest = config["ingest"]
    splitter_cfg = ingest["splitter"]
    return SentenceSplitter(
        chunk_size=splitter_cfg["chunk_size"],
        chunk_overlap=splitter_cfg["chunk_overlap"],
    )


def _extract_work_record_title(filename: str) -> str:
    stem = filename.rsplit(".", 1)[0]
    prefix = "YYYY-MM-DD-HH:MM-"
    if len(stem) > len(prefix):
        stem = stem[len(prefix):]
    suffix = "-工作记录"
    if stem.endswith(suffix):
        stem = stem[:-len(suffix)]
    return stem


class RAGKnowledgeBase:
    """
    RAG 知识库实例 (ChromaDB 持久化)

    管理两个 ChromaDB collection:
      - inspection_reports: 巡检报告，按日期检索
      - work_records: 工作记录，按标题/内容检索

    启动时通过 init_from_disk() 自动注入磁盘文件。
    """

    def __init__(self):
        self._embeddings = {}
        self._indexes = {}
        self._collections = {}
        self._init_done = False

    def _lazy_init(self):
        if self._init_done:
            return
        import chromadb
        from llama_index.vector_stores.chroma import ChromaVectorStore
        from utils.llm import create_embeddings

        VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

        self._embeddings = create_embeddings()

        db = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))

        for kb_name in ("inspection_reports", "work_records"):
            collection = db.get_or_create_collection(kb_name)
            self._collections[kb_name] = collection
            vector_store = ChromaVectorStore(chroma_collection=collection)
            self._indexes[kb_name] = VectorStoreIndex.from_vector_store(
                vector_store,
                embed_model=self._embeddings[kb_name],
            )

        self._init_done = True

    def _get_retrieve_config(self, kb_name: str) -> dict:
        return get_embedding_config(kb_name)["retrieve"]

    def _file_fingerprint(self, filepath: Path) -> str:
        stat = filepath.stat()
        raw = f"{filepath.name}|{stat.st_size}|{stat.st_mtime}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _get_ingested_fingerprints(self, kb_name: str) -> set:
        collection = self._collections[kb_name]
        existing = collection.get(include=["metadatas"])
        fps = set()
        if existing and existing["metadatas"]:
            for meta in existing["metadatas"]:
                if meta and "file_fingerprint" in meta:
                    fps.add(meta["file_fingerprint"])
        return fps

    # ============================================================
    # 启动时注入磁盘文件
    # ============================================================

    async def init_from_disk(self):
        """
        扫描 inspection_reports/ 和 work_records/ 目录，
        将尚未被注入（或已变更）的文件加载到向量库。

        通过 file_fingerprint (文件名+大小+修改时间 的 MD5) 去重。
        应在应用启动时调用一次。
        """
        self._lazy_init()

        await self._ingest_directory(
            kb_name="inspection_reports",
            directory=INSPECTOR_REPORTS_DIR,
            glob_pattern="*.md",
            build_metadata=lambda path: {
                "file_fingerprint": self._file_fingerprint(path),
                "type": "inspection",
                "filename": path.name,
                "date": f"{path.stem[:10]} {path.stem[11:13]}:{path.stem[14:16]}",
            },
        )

        await self._ingest_directory(
            kb_name="work_records",
            directory=WORK_RECORDS_DIR,
            glob_pattern="*.md",
            build_metadata=lambda path: {
                "file_fingerprint": self._file_fingerprint(path),
                "type": "work_record",
                "filename": path.name,
                "title": _extract_work_record_title(path.name),
            },
        )

    async def _ingest_directory(
        self,
        kb_name: str,
        directory: Path,
        glob_pattern: str,
        build_metadata: Callable[[Path], dict],
    ):
        if not directory.exists():
            _rag_log.warning(f"目录不存在: {directory}")
            return

        existing_fps = self._get_ingested_fingerprints(kb_name)
        splitter = _splitter_from_config(kb_name)

        files = sorted(directory.glob(glob_pattern))
        ingested_count = 0

        for filepath in files:
            fp = self._file_fingerprint(filepath)
            if fp in existing_fps:
                continue

            content = filepath.read_text(encoding="utf-8")
            if not content.strip():
                continue

            metadata = build_metadata(filepath)

            doc = Document(text=content, metadata=metadata)
            nodes = splitter.get_nodes_from_documents([doc])
            for node in nodes:
                node.metadata.update(metadata)

            enter_write()
            try:
                self._indexes[kb_name].insert_nodes(nodes)
            finally:
                exit_write()
            ingested_count += 1
            _rag_log.info(f"注入 [{kb_name}] {filepath.name} ({len(content)} chars)")
        _rag_log.info(f"注入完成 [{kb_name}] 新增 {ingested_count} 个文件")

    # ============================================================
    # 巡检报告 (inspection_reports) — 注入 + 检索
    # ============================================================

    async def ingest_inspection(self, content: str, metadata: Optional[dict] = None):
        self._lazy_init()
        kb = "inspection_reports"
        meta = metadata or {}
        meta.setdefault("type", "inspection")
        meta.setdefault("ingested_at", datetime.now().isoformat())

        length = len(content)
        _rag_log.info(
            f"写入知识库 [{kb}] "
            f"content_length={length} metadata={json.dumps(meta, ensure_ascii=False)}"
        )

        splitter = _splitter_from_config(kb)
        doc = Document(text=content, metadata=meta)
        nodes = splitter.get_nodes_from_documents([doc])
        for node in nodes:
            node.metadata.update(meta)

        enter_write()
        try:
            self._indexes[kb].insert_nodes(nodes)
        finally:
            exit_write()

    async def retrieve_inspection(
        self,
        query: str,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> list:
        self._lazy_init()
        kb = "inspection_reports"
        retrieve_cfg = self._get_retrieve_config(kb)

        retriever = self._indexes[kb].as_retriever(
            similarity_top_k=retrieve_cfg["similarity_top_k"],
        )

        nodes = await retriever.aretrieve(query)
        results = []
        for node in nodes:
            if node.score is not None and node.score < retrieve_cfg["similarity_cutoff"]:
                continue
            meta = node.metadata or {}
            doc_date = meta.get("date", "")
            if date_from and doc_date < date_from:
                continue
            if date_to and doc_date > date_to:
                continue
            results.append({
                "content": node.get_content(),
                "score": node.score,
                "metadata": meta,
            })

        count = len(results)
        _rag_log.info(f"从 [{kb}] 检索 query='{query[:60]}...' 返回 {count} 条结果")
        return results

    # ============================================================
    # 工作记录 (work_records) — 注入 + 检索
    # ============================================================

    async def ingest_work_record(self, content: str, metadata: Optional[dict] = None):
        self._lazy_init()
        kb = "work_records"
        meta = metadata or {}
        meta.setdefault("type", "work_record")
        meta.setdefault("ingested_at", datetime.now().isoformat())

        length = len(content)
        _rag_log.info(
            f"写入知识库 [{kb}] "
            f"content_length={length} metadata={json.dumps(meta, ensure_ascii=False)}"
        )

        splitter = _splitter_from_config(kb)
        doc = Document(text=content, metadata=meta)
        nodes = splitter.get_nodes_from_documents([doc])
        for node in nodes:
            node.metadata.update(meta)

        enter_write()
        try:
            self._indexes[kb].insert_nodes(nodes)
        finally:
            exit_write()

    async def retrieve_work_record(
        self,
        query: str,
        title_keyword: Optional[str] = None,
    ) -> list:
        self._lazy_init()
        kb = "work_records"
        retrieve_cfg = self._get_retrieve_config(kb)

        retriever = self._indexes[kb].as_retriever(
            similarity_top_k=retrieve_cfg["similarity_top_k"],
        )

        nodes = await retriever.aretrieve(query)
        results = []
        for node in nodes:
            if node.score is not None and node.score < retrieve_cfg["similarity_cutoff"]:
                continue
            meta = node.metadata or {}
            if title_keyword:
                doc_title = meta.get("title", "")
                if title_keyword not in doc_title:
                    continue
            results.append({
                "content": node.get_content(),
                "score": node.score,
                "metadata": meta,
            })

        count = len(results)
        _rag_log.info(f"从 [{kb}] 检索 query='{query[:60]}...' 返回 {count} 条结果")
        return results

    # ============================================================
    # 统计查询
    # ============================================================

    def get_inspection_count(self) -> int:
        self._lazy_init()
        return self._collections["inspection_reports"].count()

    def get_work_record_count(self) -> int:
        self._lazy_init()
        return self._collections["work_records"].count()
