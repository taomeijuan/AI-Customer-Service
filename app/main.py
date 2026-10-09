from contextlib import asynccontextmanager
from pathlib import Path

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
# 静默第三方 HTTP 客户端：httpx 每次 LLM/rerank 调用都打 INFO，serve.sh 终端刷屏
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpx2").setLevel(logging.WARNING)

# ch07 上下文可观测：同一条 INFO 流双写 stderr + log/app.log（model_ctx/history_ctx 可 grep）
_LOG_DIR = Path(__file__).resolve().parents[1] / "log"
_LOG_DIR.mkdir(exist_ok=True)
logging.getLogger().addHandler(
    logging.FileHandler(_LOG_DIR / "app.log", encoding="utf-8")
)

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.chat import router as chat_router
from app.api.conversations import router as conversations_router
from app.api.extract import router as extract_router
from app.api.rag_eval import router as rag_eval_router
from app.api.refunds import router as refunds_router
from app.api.tickets import router as tickets_router
from app.core.config import get_settings
from app.core.llm import get_chat_model
from app.db.engine import build_engine
from app.extraction.schemas import AfterSalesExtraction
from app.extraction.service import ExtractionService, build_structured_model
from app.generation.answerer import AnswerSchema, Answerer
from langgraph.checkpoint.memory import InMemorySaver
from fastapi.responses import FileResponse
from app.knowledge.embedder import build_embedder
from app.knowledge.query_rewriter import LangChainRewriter
from app.knowledge.reranker import build_reranker
from app.knowledge.milvus_store import LazyMilvusStore
from app.knowledge.retriever import HybridRetriever
from app.workflow.agent_node import build_agent_node
from app.workflow.expander import ExpansionSchema, Expander
from app.workflow.graph import build_workflow
from app.workflow.intent import LangChainIntentClassifier
from app.workflow.refund_flow import build_refund_prep
from app.workflow.resolver import ResolutionSchema, Resolver
from app.tools.ecommerce import query_logistics, query_order, query_product
from app.tools.ticket import build_create_ticket_tool

STATIC_DIR = Path(__file__).parent / "static"

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        await app.state.engine.dispose()  # 进程退出时释放连接池

    app = FastAPI(title="ecom-cs", version="0.4.0", lifespan=lifespan)
    settings = get_settings()
    app.state.settings = settings
    app.state.engine, app.state.session_factory = build_engine(settings)

    # ch07 启动自检：只依赖配置，create_app 期算（ASGITransport 测试不走 lifespan 也在）
    from app.memory.budget import compute_budget

    budget = compute_budget(settings)
    app.state.context_budget = budget
    if not budget.ok:
        logger.error(
            "上下文预算不足：窗口匀出 %d、期望 %d、历史预算仅 %d < 单轮稳态 %d，"
            "请调小固定预留/步数/工具峰值或换大窗口模型",
            budget.window_avail, budget.desired, budget.history, settings.turn_steady_tokens,
        )
    else:
        logger.info(
            "context budget: history=%d layer1=%d layer2=%d (peak=%d fixed=%d)",
            budget.history, budget.layer1, budget.layer2, budget.peak, budget.fixed,
        )

    embedder = build_embedder(settings)
    milvus_store = LazyMilvusStore(
        uri=settings.milvus_uri, collection=settings.milvus_collection
    )
    app.state.embedder = embedder
    app.state.milvus_store = milvus_store

    model = get_chat_model()
    structured = build_structured_model(model)
    app.state.extract_service = ExtractionService(structured)

    retriever = HybridRetriever(
        milvus=milvus_store,
        embedder=embedder,
        rewriter=LangChainRewriter(model),
        reranker=build_reranker(settings),
        settings=settings,
    )
    app.state.retriever = retriever
    app.state.answerer = Answerer(build_structured_model(model, AnswerSchema))

    agent_node = build_agent_node(
        llm=model,
        tools=[query_order, query_product, query_logistics],  # ch02 业务工具；知识类由图检索节点承担
        settings=settings,
    )
    refund_prep = build_refund_prep(
        retriever=retriever,
        expander=Expander(build_structured_model(model, ExpansionSchema)),  # ch06 Query 扩写
    )
    from app.memory.summarizer import Summarizer

    app.state.summarizer = Summarizer(model, app.state.session_factory, settings)
    app.state.workflow = build_workflow(
        retriever=retriever,
        agent_node=agent_node,
        intent_classifier=LangChainIntentClassifier(model),
        session_factory=app.state.session_factory,
        resolver=Resolver(build_structured_model(model, ResolutionSchema)),  # ch06 指代消解+改写
        refund_prep=refund_prep,  # ch06 退款确定性子流程
        checkpointer=InMemorySaver(),
    )

    @app.get("/healthz")
    async def healthz() -> dict:
        budget = getattr(app.state, "context_budget", None)
        return {"status": "ok", "budget_ok": bool(budget and budget.ok)}

    @app.get("/rag-eval", include_in_schema=False)
    async def rag_eval_page() -> FileResponse:
        return FileResponse(STATIC_DIR / "rag_eval.html")

    app.include_router(chat_router)
    app.include_router(conversations_router)
    app.include_router(extract_router)
    app.include_router(rag_eval_router)
    app.include_router(refunds_router)
    app.include_router(tickets_router)
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

    @app.middleware("http")
    async def no_cache_front_page(request, call_next):
        """前端页面禁缓存：每次加载都取最新 js（浏览器启发式缓存曾导致旧版逻辑被杀）。
        API 响应不动——只对 HTML 页面生效。"""
        resp = await call_next(request)
        if request.url.path in ("/", "/index.html") and resp.status_code == 200:
            resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            resp.headers["Pragma"] = "no-cache"
            resp.headers["Expires"] = "0"
        return resp

    return app


app = create_app()
