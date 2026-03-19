# FlyChat External AI Agent — Setup Guide

## What this is
A standalone Python + FastAPI microservice hosted on Railway.
It replaces the OpenAI calls inside your Replit backend with a dedicated,
language-aware AI agent that always returns ONE reply + ONE silent DB action.

---

## STEP 1 — Create GitHub repo for the agent

1. Go to github.com → New repository → name it `flychat-ai-agent`
2. Push these files to it:
   - main.py
   - agent.py
   - requirements.txt
   - railway.toml

```bash
git init
git add .
git commit -m "Initial AI agent"
git remote add origin https://github.com/YOUR_USERNAME/flychat-ai-agent.git
git push -u origin main
```

---

## STEP 2 — Deploy on Railway

1. Go to railway.app → New Project → Deploy from GitHub repo
2. Select your `flychat-ai-agent` repo
3. Railway will auto-detect Python and deploy

**Set these Environment Variables in Railway:**

| Variable | Value |
|----------|-------|
| `OPENAI_API_KEY` | Your OpenAI API key |
| `AGENT_SECRET` | Any strong random string (e.g. run: `openssl rand -hex 32`) |

4. After deploy, copy your Railway URL (e.g. `https://flychat-ai-agent-production.up.railway.app`)
5. Test it: `GET https://your-url.railway.app/health` → should return `{"status":"ok"}`

---

## STEP 3 — Add secrets to Replit

In Replit → Tools → Secrets, add:

| Key | Value |
|-----|-------|
| `AI_AGENT_URL` | Your Railway URL (no trailing slash) |
| `AGENT_SECRET` | Same value you set in Railway |
| `JWT_SECRET` | Run `openssl rand -hex 32` and paste result |

---

## STEP 4 — Add the bridge file to FlyChat backend

1. Copy `ai-agent-bridge.ts` to:
   `artifacts/api-server/src/lib/ai-agent-bridge.ts`

2. In `automation-engine.ts`, find the import section at the top and add:
```typescript
import { handleAiReplyForMessage } from "./ai-agent-bridge.js";
```

3. Find the existing `handleAiReplyForMessage` function in automation-engine.ts
   and DELETE the entire function body (keep only the import above).

4. Find everywhere `handleAiReplyForMessage(...)` is called and update
   the call to pass the new params shape:

```typescript
// In automation-engine.ts, replace the existing call with:
await handleAiReplyForMessage({
  messageId: message.id,
  conversationId: conv.id,
  storeId: store.id,
  storeName: store.name,
  aiSystemPrompt: store.aiSystemPrompt ?? undefined,
  products: products,           // already loaded in your function
  recentOrders: recentOrders,  // already loaded in your function
  emitNewMessage: (convId, sId, msgId, content) => {
    // paste your existing emitNewMessage/socket emit logic here
    io.to(`conv:${convId}`).emit("new_message", { id: msgId, content, sender: "bot" });
    io.to(`store:${sId}`).emit("new_message", { id: msgId, content, conversationId: convId });
  },
  consumeCredits: () => consumeCredits(store.id),
  checkCredits: async () => {
    const status = await getAiStatus(store.id);
    return status.eligible;
  },
});
```

---

## STEP 5 — Test

1. Open the widget on your test store
2. Send "salam alikom" → AI should reply in Darija
3. Send "bonjour" → AI should reply in French
4. Ask for products → clean numbered list, not pipes
5. Complete an order flow → ONE message per turn, order appears in dashboard tagged "AI agent"
6. Ask to cancel → ONE message, order status changes in dashboard

---

## File Structure

```
flychat-ai-agent/           ← Deploy this to Railway
├── main.py                 ← FastAPI app + /chat endpoint
├── agent.py                ← Language detection + AI + extraction
├── requirements.txt
└── railway.toml

artifacts/api-server/src/lib/
└── ai-agent-bridge.ts      ← Add this to Replit (bridge to Railway)
```

---

## How it works (flow)

```
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
```
