# CBAMForge — AI-Powered EU CBAM Compliance Agent
**Google Cloud AI Builder Cup 2026 | Manufacturing Track**

### Live Deployed Prototype
- **Frontend Dashboard (Firebase Hosting):** https://argmax-ai-builder.web.app
- **Backend API (Google Cloud Run):** https://builder-cup-api-loxbvxjoya-uc.a.run.app/docs

### Tech Stack
- **Firebase Hosting:** Zero-cold-start web dashboard (`public/index.html`)
- **Google Cloud Run:** Containerized Python FastAPI service (`backend/main.py`, `backend/Dockerfile`)
- **Cloud Firestore:** Real-time audit persistence (`cbam_audits`) & 5,000-request safety circuit breaker (`system_guardrails`)
- **Gemini 2.5 Flash:** Multimodal regulatory calculation and EU CBAM XML generation
