# Implementation Plan: Dashboard Refresh Button + Chatbot Web Search

## Overview

Two independent features:

1. **Refresh Button** — A button on `dashboard.html` that triggers `run_screener.py` + `generate_dashboard.py` + `deploy_dashboard.sh` on the local machine, then pushes to GitHub so all users with the GitHub Pages link see fresh data.

2. **Chatbot Web Search** — Upgrade the existing OpenAI-based chatbot to use Claude (Anthropic API) with web search capability + full backend data context, so the assistant can explain why any stock was rated the way it was AND go deeper using live web search.

---

## Feature 1: Refresh Button

### Problem Statement

Currently, refreshing the dashboard is a manual 3-step process:
1. `python run_screener.py`
2. `python generate_dashboard.py`
3. `bash deploy_dashboard.sh`

Users with the GitHub Pages link (`https://calebsmit.github.io/screener-dashboard/`) only see stale data until Caleb manually runs these steps.

### Architecture Decision

**Option A — Pure Browser Button (Rejected)**
A button in `dashboard.html` cannot shell-execute Python scripts on the server. Not viable for a static site.

**Option B — Local Flask/FastAPI Server + Button (Chosen)**
Add a tiny local REST server (`refresh_server.py`) that:
- Listens on `localhost:7720`
- Exposes `POST /refresh` endpoint
- Runs the pipeline + deploy script as a subprocess
- Streams progress back via Server-Sent Events (SSE)

The dashboard button calls `http://localhost:7720/refresh`. If the server is not running, the button degrades gracefully with a tooltip: _"Start refresh_server.py to enable"_.

**Why not GitHub Actions?**
The screener fetches live market data from yfinance (requires internet, takes ~10–20 min). GitHub Actions runners could do this but would require secrets management and a scheduled trigger — that's Phase 2. For now, the local machine is the data source; the server just automates the deploy.

### Implementation Steps

#### Step 1 — Create `refresh_server.py`

New file in project root. ~100 lines.

```python
# refresh_server.py
from http.server import HTTPServer, BaseHTTPRequestHandler
import subprocess, threading, json, os, sys

PORT = 7720
PIPELINE = [sys.executable, "run_screener.py"]
DASHBOARD = [sys.executable, "generate_dashboard.py"]
DEPLOY    = ["bash", "deploy_dashboard.sh"]

_lock = threading.Lock()

class Handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self._cors()
        self.end_headers()

    def do_POST(self):
        if self.path != "/refresh":
            self.send_response(404); self.end_headers(); return
        if not _lock.acquire(blocking=False):
            self.send_response(409)
            self._cors()
            self.end_headers()
            self.wfile.write(b'{"error":"refresh already running"}')
            return

        self.send_response(200)
        self._cors()
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        try:
            for step, cmd in [("screener", PIPELINE), ("dashboard", DASHBOARD), ("deploy", DEPLOY)]:
                self._send(f"step:{step}")
                result = subprocess.run(
                    cmd, capture_output=True, text=True,
                    cwd=os.path.dirname(os.path.abspath(__file__))
                )
                if result.returncode != 0:
                    self._send(f"error:{step}:{result.stderr[-500:]}")
                    return
                self._send(f"done:{step}")
            self._send("complete")
        finally:
            _lock.release()

    def _send(self, data):
        try:
            self.wfile.write(f"data: {data}\n\n".encode())
            self.wfile.flush()
        except BrokenPipeError:
            pass

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "null")  # file:// origin
        self.send_header("Access-Control-Allow-Origin", "*")

    def log_message(self, *args): pass  # silence access logs

if __name__ == "__main__":
    print(f"Refresh server running on http://localhost:{PORT}")
    print("Press Ctrl+C to stop.")
    HTTPServer(("localhost", PORT), Handler).serve_forever()
```

**Key design choices:**
- `threading.Lock` prevents concurrent refresh runs (returns 409 if busy)
- SSE (Server-Sent Events) streams progress without WebSocket complexity
- Runs each step sequentially, stops on error
- No external dependencies (stdlib only)
- CORS allows `null` origin (when dashboard is opened as `file://`)

#### Step 2 — Add Refresh UI to `dashboard.html`

In `generate_dashboard.py`, find the `generate_html()` function (line ~683) which builds the HTML string. Two changes:

**2a. Add the refresh button to the header bar** (near the existing "Last Updated" timestamp, around line ~750 in the HTML template string):

```html
<!-- Refresh Button -->
<button id="refresh-btn" onclick="triggerRefresh()" 
  title="Requires refresh_server.py running locally"
  style="...">
  ↻ Refresh Data
</button>
<div id="refresh-status" style="display:none; font-size:12px; color:var(--accent)"></div>
```

**2b. Add `triggerRefresh()` JavaScript** (near the bottom of the `<script>` block):

```javascript
async function triggerRefresh() {
    const btn = document.getElementById('refresh-btn');
    const status = document.getElementById('refresh-status');
    const STEPS = { screener: 'Running screener (10-20 min)...', 
                    dashboard: 'Generating dashboard...', 
                    deploy: 'Deploying to GitHub...' };
    
    // Check server is reachable first
    try {
        await fetch('http://localhost:7720/ping', { signal: AbortSignal.timeout(1000) });
    } catch {
        alert('Refresh server not running.\n\nStart it with:\n  python refresh_server.py\n\nThen try again.');
        return;
    }

    btn.disabled = true;
    btn.textContent = '⟳ Starting...';
    status.style.display = 'block';

    const es = new EventSource('http://localhost:7720/refresh-sse');  
    // Note: POST via EventSource not supported; switch server to GET /refresh-sse
    
    es.onmessage = function(e) {
        const msg = e.data;
        if (msg.startsWith('step:')) {
            const step = msg.split(':')[1];
            status.textContent = STEPS[step] || step;
        } else if (msg === 'complete') {
            status.textContent = '✓ Deployed! Reloading in 5s...';
            es.close();
            setTimeout(() => location.reload(), 5000);
        } else if (msg.startsWith('error:')) {
            status.textContent = '✗ Error: ' + msg.split(':').slice(2).join(':');
            btn.disabled = false;
            btn.textContent = '↻ Refresh Data';
            es.close();
        }
    };
    es.onerror = function() {
        status.textContent = 'Server disconnected.';
        btn.disabled = false; btn.textContent = '↻ Refresh Data';
        es.close();
    };
}
```

**Note on EventSource vs POST:** EventSource only does GET. Two options:
- Switch server endpoint to GET `/refresh-sse` (start refresh + stream in same GET)  
- Or: POST `/refresh/start`, get a job ID, then GET `/refresh/stream?id=...`

For simplicity: use GET `/refresh-sse` — idempotent enough since the lock prevents double runs.

#### Step 3 — Add `/ping` endpoint to `refresh_server.py`

```python
def do_GET(self):
    if self.path == "/ping":
        self.send_response(200); self._cors(); self.end_headers()
        self.wfile.write(b'ok')
    elif self.path == "/refresh-sse":
        # Same logic as POST /refresh above, just GET
        ...
```

Refactor server to handle both GET `/ping` and GET `/refresh-sse`.

#### Step 4 — Update `run_screener.py` skill

Update `.claude/commands/run-screener.md` to mention `refresh_server.py`:
```
python refresh_server.py   # Start the refresh server (keep running)
```

#### Step 5 — Document in README

Add a "Live Refresh" section explaining:
- Start `python refresh_server.py` once (keep terminal open)
- Click "↻ Refresh Data" in dashboard
- ~20 min later dashboard auto-reloads and GitHub Pages updates

### Key Files to Modify

| File | Operation | Description |
|------|-----------|-------------|
| `refresh_server.py` | Create | New local REST server (~120 lines) |
| `generate_dashboard.py` | Modify | Add refresh button HTML + JS to template |
| `dashboard.html` | Regenerated | Auto-updated when generate_dashboard.py runs |

### Risks and Mitigation

| Risk | Mitigation |
|------|------------|
| Server not running → button does nothing | Ping check before refresh; show alert with instructions |
| Screener takes 20+ min → user closes tab | SSE reconnects; server continues running regardless of browser |
| Concurrent refresh clicks | Server-side lock returns 409; button disabled while running |
| Git push fails (no credentials) | Error surfaced in SSE stream |
| Windows `bash` not found for deploy script | Detect OS in server; use `subprocess` with `shell=True` on Windows |

---

## Feature 2: Chatbot Web Search (Claude + Brave/Tavily)

### Problem Statement

The current chatbot uses OpenAI (GPT-4o-mini) with:
- Static screener data embedded in system prompt
- No web access
- User must supply their own OpenAI key

Goal: Switch to **Claude** (best for reasoning + tool use) with:
- Same screener data context (why a stock scored what it did)
- **Web search** via Anthropic's built-in web search tool (as of Claude 3.5+)
- Deeper stock/industry research capability

### Architecture Decision

**Claude API with `web_search_20250305` tool (Chosen)**

Anthropic now provides a built-in web search tool available via the Messages API. Usage:
```json
{
  "tools": [{"type": "web_search_20250305", "name": "web_search"}],
  "tool_choice": {"type": "auto"}
}
```

This is simpler than routing through Tavily/Brave (no extra API key). Claude decides when to search.

**Still client-side (no backend required)** — The Anthropic API is called directly from the browser using the user's Anthropic API key, stored in `localStorage`. This matches the existing architecture pattern.

**Note on CORS:** Anthropic's API does allow browser-side calls, but you need to set `dangerouslyAllowBrowser: true` with the SDK, or call the raw REST endpoint directly. We'll call the REST endpoint directly (same as the current OpenAI approach).

### Implementation Steps

#### Step 1 — Update API Key Dialog in `generate_dashboard.py`

Find the existing API key dialog (around line 4268 in dashboard.html, generated by generate_dashboard.py). Change:
- Label: "Anthropic API Key" (was "OpenAI API Key")
- Prefix check: `sk-ant-` (was `sk-`)
- localStorage key: `screener_anthropic_api_key` (was `screener_openai_api_key`)
- Model selection: `claude-sonnet-4-6` (default), `claude-opus-4-6` (add option)
- Remove GPT-4o-mini / GPT-4o options

#### Step 2 — Rewrite `makeOpenAIChatRequest()` → `makeClaudeRequest()`

Current function (line ~4629) POSTs to `https://api.openai.com/v1/chat/completions`.

New function POSTs to `https://api.anthropic.com/v1/messages` with:

```javascript
async function makeClaudeRequest(userMessage, userContext) {
    const contextMsg = userContext 
        ? '[SCREENER DATA]\n' + userContext + '\n\n[QUESTION]\n' + userMessage 
        : userMessage;

    const body = {
        model: getChatModel(),  // 'claude-sonnet-4-6' default
        max_tokens: 1024,
        system: buildSystemPrompt(),
        tools: [{
            type: "web_search_20250305",
            name: "web_search",
            max_uses: 3  // limit cost
        }],
        messages: [
            ...chatState.messages.slice(-chatState.maxHistory).map(m => ({
                role: m.role === 'ai' ? 'assistant' : m.role,
                content: m.content
            })),
            { role: 'user', content: contextMsg }
        ]
    };

    const res = await fetch('https://api.anthropic.com/v1/messages', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'x-api-key': getApiKey(),
            'anthropic-version': '2023-06-01',
            'anthropic-beta': 'interleaved-thinking-2025-05-14',  // optional
        },
        body: JSON.stringify(body)
    });

    if (!res.ok) {
        const err = await res.json();
        throw new Error(err.error?.message || 'Anthropic API error ' + res.status);
    }

    const data = await res.json();
    
    // Parse response — Claude may return text blocks + tool_use blocks
    let fullText = '';
    let searchedFor = [];
    
    for (const block of data.content) {
        if (block.type === 'text') {
            fullText += block.text;
        } else if (block.type === 'tool_use' && block.name === 'web_search') {
            searchedFor.push(block.input.query);
        }
    }
    
    // Show search queries as a subtle footnote
    if (searchedFor.length > 0) {
        fullText += '\n\n---\n*Searched: ' + searchedFor.map(q => '`' + q + '`').join(', ') + '*';
    }
    
    return fullText;
}
```

**Note on streaming:** The Anthropic streaming API uses `anthropic-beta: "stream-2025-xx-xx"` with SSE. For the first implementation, non-streaming (wait for full response) is simpler and works fine for web search (which has latency anyway). Streaming can be added in Phase 2.

#### Step 3 — Update `buildSystemPrompt()` for Claude + Web Search

Add to the existing system prompt:

```
## Web Search Capability
You have access to real-time web search. Use it when:
- Asked about recent news, earnings, analyst ratings, or events for a specific stock
- Asked about an industry trend or macro factor
- The user wants to go deeper than what the screener data shows

When searching, be precise: search "[TICKER] Q4 2025 earnings", "[TICKER] analyst price target 2025", "[INDUSTRY] sector outlook 2025", etc.

Always ground your search-based answers with screener scores. For example: "Apple scores 72/100 on Quality. Recent web search shows [finding], which supports/contradicts this because..."

Do NOT search for general screener methodology questions — those are answered from the embedded data.
```

#### Step 4 — Update Starter Questions

Change the starter questions to reflect Claude + web search capability:

```javascript
const SUGGESTIONS = [
    "Why is AAPL ranked #1? Search for recent news.",
    "Which sector looks strongest? Compare to recent trends.",  
    "Why was NVDA excluded from the portfolio?",
    "Search for latest analyst ratings on our top 5 holdings.",
    "What's driving the low valuation scores this quarter?",
    "Find recent earnings surprises for our portfolio stocks."
];
```

#### Step 5 — Handle Tool-Use Multi-Turn (Advanced)

The Anthropic API with web search works in a multi-turn loop:
1. Send user message
2. Claude responds with `tool_use` blocks (search queries)
3. Client must send `tool_result` blocks back
4. Claude synthesizes final answer

This is more complex than step 2 above. The proper implementation:

```javascript
async function makeClaudeRequest(userMessage, userContext) {
    const messages = [...buildMessages(userMessage, userContext)];
    
    while (true) {
        const res = await fetch('https://api.anthropic.com/v1/messages', {
            method: 'POST',
            headers: { ... },
            body: JSON.stringify({ model, max_tokens: 1024, system, tools, messages })
        });
        const data = await res.json();
        
        if (data.stop_reason === 'end_turn') {
            // Extract text content
            return data.content.filter(b => b.type === 'text').map(b => b.text).join('');
        }
        
        if (data.stop_reason === 'tool_use') {
            // Add Claude's response to messages
            messages.push({ role: 'assistant', content: data.content });
            
            // Web search is handled server-side by Anthropic — 
            // the tool results come back in the NEXT response automatically
            // (when using type: "web_search_20250305", Anthropic executes the search)
            // So we just need to continue the loop
            
            // Show "Searching..." indicator
            updateTypingIndicator('Searching the web...');
        }
    }
}
```

**Important clarification:** With `type: "web_search_20250305"`, Anthropic's server executes the search internally. The client does NOT need to provide search results back. Claude handles it server-side and continues generating. This means the multi-turn loop is simpler — just keep calling until `stop_reason === 'end_turn'`.

#### Step 6 — UI: Show Search Activity

Add a subtle indicator when web search is in progress:

```javascript
// In the typing indicator, cycle through:
// "Thinking..." → "Searching the web..." → "Analyzing results..." → "Writing response..."
```

### Key Files to Modify

| File | Operation | Description |
|------|-----------|-------------|
| `generate_dashboard.py` | Modify | Replace OpenAI chat code with Claude API code (~150 lines in template) |
| `dashboard.html` | Regenerated | Auto-updated |

### Risks and Mitigation

| Risk | Mitigation |
|------|------------|
| CORS blocked for Anthropic API in browser | Call raw REST (not SDK); Anthropic allows browser calls with proper headers |
| User has no Anthropic key | Dialog explains what key is needed and links to console.anthropic.com |
| Web search cost (Anthropic charges per search) | `max_uses: 3` cap per message; show searches used in footnote |
| `web_search_20250305` tool availability | Check Anthropic docs — may require beta header; degrade gracefully if 400 |
| Non-streaming response feels slow | Show typing indicator immediately; add streaming in Phase 2 |

---

## Implementation Order

### Phase 1 (Do First — Independent)
1. Create `refresh_server.py`
2. Add refresh button + JS to `generate_dashboard.py` template
3. Test locally: run server, click button, verify deploy

### Phase 2 (Do Second — Independent of Phase 1)
4. Rewrite chat API call in `generate_dashboard.py` to use Claude
5. Update API key dialog (Anthropic key)
6. Update system prompt for web search guidance
7. Update starter questions
8. Test: open dashboard, enter Anthropic key, ask question that triggers search

### Phase 3 (Polish)
9. Add streaming to Claude responses
10. Consider GitHub Actions scheduled run (replaces need for local refresh server)

---

## SESSION_ID
- CODEX_SESSION: N/A (plan generated by Claude directly)
- GEMINI_SESSION: N/A
