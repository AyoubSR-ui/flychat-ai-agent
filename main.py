from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
from agent import process_message

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
    stock: int
    variants: Optional[list] = None
    imageUrl: Optional[str] = None
    description: Optional[str] = None


class RecentOrder(BaseModel):
    id: str
    orderNumber: str
    status: str
    customerName: Optional[str] = None
    customerPhone: Optional[str] = None


class ChatRequest(BaseModel):
    conversationId: str
    storeId: str
    storeName: str
    aiSystemPrompt: Optional[str] = None
    history: list[Message]
    products: list[Product]
    recentOrders: list[RecentOrder]
    aiFlowState: Optional[str] = None
    detectedLanguage: Optional[str] = None
    shippingOptions: Optional[dict] = None


class OrderItem(BaseModel):
    productId: Optional[str] = None
    productName: str
    price: float = 0.0
    quantity: int = 1
    variant: Optional[str] = None


class OrderAction(BaseModel):
    type: str  # "create_order" | "cancel_order" | "none"
    customerName: Optional[str] = None
    customerPhone: Optional[str] = None
    wilaya: Optional[str] = None
    address: Optional[str] = None
    items: Optional[list[OrderItem]] = None


class ChatResponse(BaseModel):
    reply: str
    detectedLanguage: str
    action: OrderAction


@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "selective-context-v3-retry",
        "updated": "2026-04-11"
    }

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
