from fastapi import FastAPI, HTTPException, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
from agent import process_message
from communication_optimizer import (
    estimate_optimizer_run,
    run_optimization_pipeline,
    approve_improvements,
    get_optimizer_status,
    get_behavior_improvement_layer,
)

app = FastAPI(title="FlyChat AI Agent", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

AGENT_SECRET = os.getenv("AGENT_SECRET", "")


class Message(BaseModel):
    role: str  # "customer" | "agent" | "bot"
    content: str


class Product(BaseModel):
    id: str
    name: str
    price: float
    stock: Optional[int] = None
    variants: Optional[list] = None
    imageUrl: Optional[str] = None
    images: Optional[list[str]] = None   # v2: array of image URLs
    description: Optional[str] = None


class RecentOrder(BaseModel):
    id: str
    orderNumber: str
    status: str
    customerName: Optional[str] = None
    customerPhone: Optional[str] = None


# ── v2 nested context objects ─────────────────────────────────────────────────
class StoreContext(BaseModel):
    name: Optional[str] = None
    persona: Optional[str] = None
    aiRules: Optional[str] = None
    language: Optional[str] = None


class ConversationContext(BaseModel):
    id: Optional[str] = None
    channel: Optional[str] = None
    ad_ref: Optional[str] = None
    intent_level: Optional[str] = None    # "hot" | "warm" | "cold"
    lead_stage: Optional[str] = None
    customer_type: Optional[str] = None   # "new" | "returning"


class ChatRequest(BaseModel):
    # ── v2 structured fields ──────────────────────────────────────────────────
    store: Optional[StoreContext] = None
    conversation: Optional[ConversationContext] = None
    shipping: Optional[dict] = None       # new format: {wilaya: {home, bureau}}
    # ── legacy fields (kept for backward compat) ─────────────────────────────
    conversationId: Optional[str] = None
    storeId: Optional[str] = None
    storeName: Optional[str] = None
    aiSystemPrompt: Optional[str] = None
    history: list[Message]
    products: list[Product]
    recentOrders: Optional[list[RecentOrder]] = None
    aiFlowState: Optional[str] = None
    detectedLanguage: Optional[str] = None
    shippingOptions: Optional[dict] = None
    imageUrl: Optional[str] = None
    imageAccessToken: Optional[str] = None


class OrderItem(BaseModel):
    productId: Optional[str] = None
    productName: str
    price: float = 0.0
    quantity: int = 1
    variant: Optional[str] = None


class OrderAction(BaseModel):
    type: str  # "create_order" | "cancel_order" | "update_order" | "escalate_human" | "none"
    customerName: Optional[str] = None
    customerPhone: Optional[str] = None
    wilaya: Optional[str] = None
    address: Optional[str] = None
    items: Optional[list[OrderItem]] = None
    reason: Optional[str] = None         # for escalate_human
    updateData: Optional[dict] = None    # for update_order


class ChatResponse(BaseModel):
    reply: Optional[str] = None          # None when escalating to human
    detectedLanguage: str
    action: OrderAction


@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "selective-context-v4-optimizer",
        "updated": "2026-04-18",
    }


# ─── Communication Optimizer Endpoints ───────────────────────────────────────

@app.post("/optimize/estimate")
async def estimate_endpoint(request: dict):
    """
    Called first — estimates cost before any API calls or credit deduction.
    Returns credits_required vs available so the UI can gate the run.
    """
    conversations = request.get("conversations", [])
    store_id = request.get("storeId")
    plan = request.get("plan", "starter")

    if not conversations:
        return {"error": "No conversations provided", "ready_to_run": False}

    estimate = estimate_optimizer_run(conversations, store_id, plan)
    return estimate


@app.post("/optimize")
async def trigger_optimization(request: dict):
    """
    Runs full pipeline. Blocked and returns billing info if insufficient credits.
    """
    conversations = request.get("conversations", [])
    store_id = request.get("storeId")
    plan = request.get("plan", "starter")
    auto_approve = request.get("autoApprove", False)

    if not conversations:
        return {"error": "No conversation data provided"}

    result = run_optimization_pipeline(conversations, store_id, plan, auto_approve)
    return result


@app.post("/optimize/approve")
async def approve_endpoint(request: dict):
    """Called from dashboard when owner approves improvements."""
    store_id = request.get("storeId")
    success = approve_improvements(store_id)
    return {"approved": success}


@app.get("/optimize/status")
async def optimizer_status_endpoint(storeId: Optional[str] = Query(default=None)):
    """Returns optimizer status for dashboard display."""
    return get_optimizer_status(storeId)

@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    x_agent_secret: Optional[str] = Header(None)
):
    if AGENT_SECRET and x_agent_secret != AGENT_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        result = await process_message(request)
        return result
    except Exception as e:
        print(f"[ERROR] /chat failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
