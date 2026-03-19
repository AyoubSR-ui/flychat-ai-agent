File Structure
flychat-ai-agent/           ← Deploy this to Railway
├── main.py                 ← FastAPI app + /chat endpoint
├── agent.py                ← Language detection + AI + extraction
├── requirements.txt
└── railway.toml

artifacts/api-server/src/lib/
└── ai-agent-bridge.ts      ← Add this to Replit (bridge to Railway)

How it works (flow)
Customer sends message
       ↓
FlyChat backend receives it
       ↓
Calls POST https://railway-url/chat with:
  - Full conversation history
  - Products catalog
  - Recent orders
  - Current flow state
       ↓
Railway AI Agent:
  1. Detects language (lingua library — accurate for Arabic/French/English/Darija)
  2. Builds system prompt
  3. Calls OpenAI → gets reply
  4. Calls OpenAI again (JSON mode) → extracts order/cancel intent
  5. Returns: { reply, detectedLanguage, action }
       ↓
FlyChat backend:
  1. Saves reply to DB
  2. Emits ONE socket message to customer
  3. Executes action silently (create/cancel order in DB)
  4. NO second message ever
