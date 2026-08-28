# ONNM System Migration Plan

## ROLE AND MISSION
You are an expert full-stack engineer and AI architect. Your task is to migrate the `OsteoNeuralNetwork-Model` (ONNM) from its current monolithic Streamlit architecture to a modern, decoupled, zero-cost serverless architecture.

The primary goals are to achieve 24/7 uptime, eliminate Streamlit watermarks, drastically improve page load and inference speeds, and keep running costs at exactly $0, while adhering to strict medical and privacy constraints.

## REFERENCE AND CONSTRAINTS
Before beginning, you must review the safety boundaries, privacy rules, and product scope documented in:
- `REDESIGN_BRIEF.md` (specifically Section 1: Non-Negotiable Invariants and Section 3B: Data and privacy).
- `overview.md` (for general product scope, hardware/staged deployment guidelines, and ethical guardrails).

**Core Invariants:**
- **Zero Data Loss:** The existing Cloudflare D1 database and user accounts must remain completely intact. Do not drop production tables.
- **Strict Privacy:** Maintain the exact Geolocation constraints (no IP logging, no exact coordinates, country-level coarsening only).
- **Zero Running Cost:** Do not introduce any paid services, third-party API keys that require credit cards, or Cloudflare paid features.

## THE TARGET ARCHITECTURE

### 1. Frontend: Cloudflare Pages
- The UI will be completely rewritten from Streamlit to standard web technologies (HTML/CSS/JS or a lightweight framework like React/Vue).
- Must be hosted on Cloudflare Pages (100% free, global CDN, 24/7 uptime).
- All visual requirements from `REDESIGN_BRIEF.md` apply to this new frontend.

### 2. Backend & Database: Cloudflare Workers + D1
- Keep the existing `cloudflare/src/worker.js` and D1 database.
- The frontend will communicate with this Worker for authentication, scan history, and retrieving Globe data.

### 3. Inference Backend: The AI Model
This is where the PyTorch/MONAI model will run. You must attempt **Option B first**. If it fails or is technically unfeasible, seamlessly pivot to **Option A**.

#### 🎯 PRIMARY PATH: Option B (Cloudflare Workers AI via ONNX)
Attempt to host the model entirely on Cloudflare's edge network.
1. **Convert:** Export the ONNM PyTorch model to the `.onnx` format.
2. **Validate:** Verify the ONNX model size is within Cloudflare Workers AI BYO (Bring Your Own) model limits (currently ~2GB) and that all tensor operations are supported by Cloudflare's ONNX runtime.
3. **Deploy:** Bind the model to a Cloudflare Worker. The frontend will send the X-ray to this Worker for near-instant inference.
*Why this is preferred: It requires zero external servers, never sleeps, and unifies the entire stack on Cloudflare.*

#### 🔄 FALLBACK PATH: Option A (Hugging Face Spaces + FastAPI)
If the model cannot be converted to ONNX, relies on unsupported PyTorch ops, or exceeds Cloudflare size limits, pivot immediately to Hugging Face Spaces.
1. **API Wrapper:** Wrap the existing PyTorch inference code in a lightweight `FastAPI` server.
2. **Deploy:** Host this server on a free Hugging Face Space (provides 16GB RAM).
3. **Integration:** Update the Cloudflare Pages frontend (or Cloudflare Worker) to send the image payload securely to the Hugging Face API endpoint for inference.

---

## IMPLEMENTATION STEPS & SAFETY CONTROLS

### Step 1: Limit Monitoring & Circuit Breakers (CRITICAL)
Before deploying the new infrastructure, you must implement hard limit blocks to protect the free tiers:
- **Cloudflare Limits:** Cloudflare free tier allows 100,000 Worker requests/day. Implement a counter/check. If daily usage hits 90% (90,000), the app must gracefully disable the uploader, display a "Daily capacity reached, please try again tomorrow" message, and trigger an alert/log for the admin.
- **Storage Limits:** D1 has a 500MB free limit. Implement a cleanup job or check that monitors database size. If storage reaches 90% capacity, automatically cycle out (delete) the oldest community-uploaded data that has already been trained on, ensuring critical user data and untested submissions are preserved.

### Step 2: Inference Proof-of-Concept
- Attempt the ONNX export (Option B).
- Write a short test script to verify inference outputs match the original PyTorch model.
- If it works, integrate with Cloudflare. If it fails, document the exact failure reason in a local markdown file and begin Option A.

### Step 3: API & Worker Updates
- Update `worker.js` to handle any new routing required by the decoupled frontend.
- Ensure CORS (Cross-Origin Resource Sharing) is correctly configured so the Cloudflare Pages frontend can talk to the Worker and/or Hugging Face API securely.

### Step 4: Frontend Development
- Port the UI. Ensure the OOD (Out of Distribution) gate, Grad-CAM overlays, and probability breakdown charts are fully functional in the browser.

### Step 5: Verification & Cutover
- Run all existing tests in `tests/`.
- Ensure the transition is seamless for existing users (passwords and history must still work).
