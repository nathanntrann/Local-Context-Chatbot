/**
 * InspectAssist Embeddable Chat Widget
 *
 * Usage: Add this to any HTML page:
 *   <script src="http://INSPECT_ASSIST_HOST:8000/static/widget.js"></script>
 *
 * Optional config before the script tag:
 *   <script>window.INSPECT_ASSIST_URL = "http://192.168.1.50:8000";</script>
 */
(function () {
  "use strict";

  // Resolve the InspectAssist server URL
  const scriptTag = document.currentScript;
  const BASE_URL =
    window.INSPECT_ASSIST_URL ||
    (scriptTag && scriptTag.src
      ? scriptTag.src.replace(/\/static\/widget\.js.*$/, "")
      : "http://localhost:8000");

  // ---------- CSS ----------
  const STYLES = `
    #ia-widget-btn {
      position: fixed;
      top: 10px;
      right: 10px;
      height: 28px;
      padding: 0 10px;
      border-radius: 6px;
      background: #3d4058;
      border: 1px solid #4a4d64;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 5px;
      z-index: 99999;
      transition: background 0.15s;
    }
    #ia-widget-btn:hover {
      background: #4a4d64;
    }
    #ia-widget-btn svg { width: 14px; height: 14px; fill: #8b8fa4; }
    #ia-widget-btn span {
      font-size: 11px;
      color: #8b8fa4;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
      font-weight: 500;
    }

    #ia-widget-panel {
      position: fixed;
      top: 44px;
      right: 10px;
      width: 300px;
      height: 340px;
      background: #13151e;
      border: 1px solid #2a2d3e;
      border-radius: 8px;
      display: none;
      flex-direction: column;
      overflow: hidden;
      z-index: 99998;
      box-shadow: 0 4px 20px rgba(0,0,0,0.4);
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
      color: #d4d6e0;
      font-size: 12px;
    }
    #ia-widget-panel.open { display: flex; }

    #ia-widget-header {
      background: #1a1d27;
      padding: 6px 10px;
      border-bottom: 1px solid #2a2d3e;
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-shrink: 0;
    }
    #ia-widget-header .ia-title {
      font-weight: 600;
      font-size: 11px;
      color: #8b8fa4;
      display: flex;
      align-items: center;
      gap: 5px;
    }
    #ia-widget-header .ia-logo {
      width: 16px; height: 16px;
      background: #6c8aff;
      border-radius: 3px;
      display: flex; align-items: center; justify-content: center;
      font-size: 8px; font-weight: 700; color: white;
    }
    #ia-widget-header .ia-actions {
      display: flex; gap: 4px;
    }
    #ia-widget-header .ia-actions button {
      background: none;
      border: none;
      color: #5a5d72;
      padding: 2px 5px;
      border-radius: 3px;
      cursor: pointer;
      font-size: 10px;
    }
    #ia-widget-header .ia-actions button:hover {
      color: #d4d6e0;
      background: #242736;
    }

    #ia-widget-messages {
      flex: 1;
      overflow-y: auto;
      padding: 8px;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    #ia-widget-messages::-webkit-scrollbar { width: 4px; }
    #ia-widget-messages::-webkit-scrollbar-track { background: transparent; }
    #ia-widget-messages::-webkit-scrollbar-thumb { background: #2a2d3e; border-radius: 2px; }

    .ia-msg {
      padding: 6px 8px;
      border-radius: 6px;
      line-height: 1.4;
      font-size: 11px;
      word-wrap: break-word;
      max-width: 90%;
    }
    .ia-msg.user {
      background: #1e3a5f;
      align-self: flex-end;
      border-bottom-right-radius: 2px;
      color: #c8daf0;
    }
    .ia-msg.assistant {
      background: #1e2132;
      border: 1px solid #2a2d3e;
      align-self: flex-start;
      border-bottom-left-radius: 2px;
    }
    .ia-msg.error {
      background: rgba(255,107,107,0.08);
      border: 1px solid rgba(255,107,107,0.3);
      color: #ff6b6b;
      align-self: flex-start;
      font-size: 10px;
    }
    .ia-msg strong { color: #6c8aff; }
    .ia-msg code {
      background: #0f1117;
      padding: 1px 4px;
      border-radius: 2px;
      font-size: 10px;
    }
    .ia-msg pre {
      background: #0f1117;
      padding: 5px;
      border-radius: 4px;
      overflow-x: auto;
      margin: 4px 0;
      font-size: 10px;
    }
    .ia-msg pre code { background: none; padding: 0; }

    .ia-thinking {
      color: #5a5d72;
      font-style: italic;
      font-size: 11px;
      display: flex;
      align-items: center;
      gap: 4px;
      padding: 4px 8px;
    }
    .ia-thinking .dots span {
      animation: ia-blink 1.4s infinite both;
      font-size: 14px;
    }
    .ia-thinking .dots span:nth-child(2) { animation-delay: 0.2s; }
    .ia-thinking .dots span:nth-child(3) { animation-delay: 0.4s; }
    @keyframes ia-blink {
      0%, 80%, 100% { opacity: 0.2; }
      40% { opacity: 1; }
    }

    .ia-tool-status {
      color: #6c8aff;
      font-style: italic;
      font-size: 10px;
      display: flex;
      align-items: center;
      gap: 4px;
      padding: 4px 8px;
    }
    .ia-tool-spinner {
      width: 10px; height: 10px;
      border: 1.5px solid #2a2d3e;
      border-top-color: #6c8aff;
      border-radius: 50%;
      animation: ia-spin 0.8s linear infinite;
    }
    @keyframes ia-spin { to { transform: rotate(360deg); } }

    .ia-suggestions-bar {
      display: flex;
      flex-direction: column;
      gap: 3px;
      padding: 4px 8px 0;
    }
    .ia-suggestions-bar button {
      background: #1a1d27;
      border: 1px solid #2a2d3e;
      color: #8b8fa4;
      padding: 3px 6px;
      border-radius: 4px;
      cursor: pointer;
      font-size: 10px;
      text-align: left;
    }
    .ia-suggestions-bar button:hover {
      border-color: #4a4d64;
      color: #d4d6e0;
    }

    #ia-widget-input-area {
      border-top: 1px solid #2a2d3e;
      padding: 6px;
      display: flex;
      gap: 4px;
      flex-shrink: 0;
    }
    #ia-widget-input {
      flex: 1;
      background: #1a1d27;
      border: 1px solid #2a2d3e;
      color: #d4d6e0;
      padding: 5px 7px;
      border-radius: 4px;
      font-size: 11px;
      font-family: inherit;
      resize: none;
      min-height: 26px;
      max-height: 52px;
      line-height: 1.3;
    }
    #ia-widget-input::placeholder { color: #3d4058; }
    #ia-widget-input:focus {
      outline: none;
      border-color: #4a4d64;
    }
    #ia-widget-send {
      background: #6c8aff;
      color: white;
      border: none;
      padding: 0 8px;
      border-radius: 4px;
      font-size: 10px;
      cursor: pointer;
      white-space: nowrap;
      font-weight: 600;
    }
    #ia-widget-send:hover { background: #5a75e6; }
    #ia-widget-send:disabled { opacity: 0.4; cursor: not-allowed; }

    .ia-welcome {
      text-align: center;
      color: #5a5d72;
      margin: auto;
      padding: 12px 10px;
    }
    .ia-welcome h3 { color: #8b8fa4; margin-bottom: 4px; font-size: 12px; font-weight: 600; }
    .ia-welcome p { font-size: 10px; line-height: 1.4; margin-bottom: 8px; }
    .ia-suggestions {
      display: flex;
      flex-direction: column;
      gap: 3px;
    }
    .ia-suggestions button {
      background: #1a1d27;
      border: 1px solid #2a2d3e;
      color: #8b8fa4;
      padding: 4px 8px;
      border-radius: 4px;
      cursor: pointer;
      font-size: 10px;
      text-align: left;
    }
    .ia-suggestions button:hover {
      border-color: #4a4d64;
      color: #d4d6e0;
    }

    #ia-model-select {
      background: #1a1d27;
      border: 1px solid #2a2d3e;
      color: #8b8fa4;
      font-size: 9px;
      padding: 2px 4px;
      border-radius: 3px;
      cursor: pointer;
      max-width: 100px;
      font-family: inherit;
      outline: none;
    }
    #ia-model-select:focus { border-color: #4a4d64; }
    #ia-model-select option { background: #1a1d27; color: #d4d6e0; }
  `;

  // ---------- HTML ----------
  const CHAT_ICON =
    '<svg viewBox="0 0 24 24"><path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm0 14H5.2L4 17.2V4h16v12z"/></svg>';

  // ---------- Widget Class ----------
  let conversationId = null;
  let isOpen = false;

  function inject() {
    // Style
    const style = document.createElement("style");
    style.textContent = STYLES;
    document.head.appendChild(style);

    // Small toolbar button
    const btn = document.createElement("button");
    btn.id = "ia-widget-btn";
    btn.innerHTML = CHAT_ICON + '<span>Ask AI</span>';
    btn.title = "InspectAssist";
    btn.addEventListener("click", toggle);
    document.body.appendChild(btn);

    // Chat panel
    const panel = document.createElement("div");
    panel.id = "ia-widget-panel";
    panel.innerHTML = `
      <div id="ia-widget-header">
        <div class="ia-title"><span class="ia-logo">IA</span> <select id="ia-model-select" title="Switch model"><option>loading...</option></select></div>
        <div class="ia-actions">
          <button onclick="document.getElementById('ia-widget-messages').innerHTML = document.getElementById('ia-welcome-tpl').innerHTML; window.__ia_cid = null;" title="New chat">New</button>
          <button onclick="document.getElementById('ia-widget-panel').classList.remove('open');" title="Close">&times;</button>
        </div>
      </div>
      <div id="ia-widget-messages">
        <div class="ia-welcome" id="ia-welcome-msg">
          <h3>InspectAssist</h3>
          <p>Thermal inspection assistant.<br>Ask about your dataset, analyze images, or troubleshoot.</p>
          <div class="ia-suggestions">
            <button onclick="window.__ia_send('What does my dataset look like?')">Dataset overview</button>
            <button onclick="window.__ia_send('Are any images mislabeled?')">Check labels</button>
            <button onclick="window.__ia_send('How do I troubleshoot false positives?')">Troubleshoot</button>
          </div>
        </div>
      </div>
      <template id="ia-welcome-tpl">
        <div class="ia-welcome">
          <h3>InspectAssist</h3>
          <p>Thermal inspection assistant.<br>Ask about your dataset, analyze images, or troubleshoot.</p>
          <div class="ia-suggestions">
            <button onclick="window.__ia_send('What does my dataset look like?')">Dataset overview</button>
            <button onclick="window.__ia_send('Are any images mislabeled?')">Check labels</button>
            <button onclick="window.__ia_send('How do I troubleshoot false positives?')">Troubleshoot</button>
          </div>
        </div>
      </template>
      <div id="ia-widget-input-area">
        <textarea id="ia-widget-input" placeholder="Ask something..." rows="1"></textarea>
        <button id="ia-widget-send">Send</button>
      </div>
    `;
    document.body.appendChild(panel);

    // Event listeners
    const input = document.getElementById("ia-widget-input");
    const sendBtn = document.getElementById("ia-widget-send");

    input.addEventListener("input", function () {
      this.style.height = "auto";
      this.style.height = Math.min(this.scrollHeight, 80) + "px";
    });

    input.addEventListener("keydown", function (e) {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    });

    sendBtn.addEventListener("click", handleSend);

    // Expose for suggestion buttons
    window.__ia_send = function (text) {
      input.value = text;
      handleSend();
    };

    // Load available models into dropdown
    loadModels();
  }

  async function loadModels() {
    const sel = document.getElementById("ia-model-select");
    try {
      const res = await fetch(BASE_URL + "/api/v1/models");
      if (!res.ok) return;
      const models = await res.json();
      sel.innerHTML = "";
      models.forEach(function (m) {
        const opt = document.createElement("option");
        opt.value = m.id;
        opt.textContent = m.name;
        opt.dataset.provider = m.provider;
        if (m.active) opt.selected = true;
        sel.appendChild(opt);
      });
      sel.addEventListener("change", switchModel);
    } catch (e) {
      sel.innerHTML = "<option>offline</option>";
    }
  }

  async function switchModel() {
    const sel = document.getElementById("ia-model-select");
    const opt = sel.options[sel.selectedIndex];
    if (!opt) return;
    const provider = opt.dataset.provider;
    const model = opt.textContent;
    try {
      await fetch(BASE_URL + "/api/v1/models/switch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ provider: provider, model: model }),
      });
    } catch (e) {
      // silent fail — next chat will use whatever model is active
    }
  }

  function toggle() {
    const panel = document.getElementById("ia-widget-panel");
    isOpen = !isOpen;
    if (isOpen) {
      panel.classList.add("open");
      document.getElementById("ia-widget-input").focus();
    } else {
      panel.classList.remove("open");
    }
  }

  function addMsg(role, text) {
    const msgs = document.getElementById("ia-widget-messages");
    // Hide welcome
    const welcome = document.getElementById("ia-welcome-msg");
    if (welcome) welcome.style.display = "none";

    const div = document.createElement("div");
    div.className = "ia-msg " + role;
    if (role === "assistant") {
      div.innerHTML = renderMd(text);
    } else {
      div.textContent = text;
    }
    msgs.appendChild(div);
    msgs.scrollTop = msgs.scrollHeight;
    return div;
  }

  function addThinking() {
    const welcome = document.getElementById("ia-welcome-msg");
    if (welcome) welcome.style.display = "none";
    const msgs = document.getElementById("ia-widget-messages");
    const div = document.createElement("div");
    div.className = "ia-thinking";
    div.id = "ia-thinking";
    div.innerHTML =
      'Thinking<div class="dots"><span>.</span><span>.</span><span>.</span></div>';
    msgs.appendChild(div);
    msgs.scrollTop = msgs.scrollHeight;
  }

  function removeThinking() {
    const el = document.getElementById("ia-thinking");
    if (el) el.remove();
  }

  function renderMd(text) {
    return text
      .replace(/```(\w*)\n([\s\S]*?)```/g, "<pre><code>$2</code></pre>")
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/\*(.+?)\*/g, "<em>$1</em>")
      .replace(/^#### (.+)$/gm, "<h4>$1</h4>")
      .replace(/^### (.+)$/gm, "<h3>$1</h3>")
      .replace(/^## (.+)$/gm, "<h2>$1</h2>")
      .replace(/^# (.+)$/gm, "<h1>$1</h1>")
      .replace(/^- (.+)$/gm, "<li>$1</li>")
      .replace(/^\d+\. (.+)$/gm, "<li>$1</li>")
      .replace(/\n/g, "<br>");
  }

  // Tool name → human label
  const TOOL_LABELS = {
    search_knowledge: "Searching knowledge base",
    search_knowledge_filtered: "Searching knowledge base",
    get_article_section: "Reading article",
    explain_concept: "Looking up concept",
    analyze_image: "Analyzing image",
    compare_images: "Comparing images",
    find_suspicious_labels: "Auditing labels",
    generate_audit_report: "Generating audit report",
    get_dataset_summary: "Loading dataset summary",
    get_dataset_statistics: "Calculating statistics",
    get_sample_images: "Fetching sample images",
  };

  function updateThinkingTool(toolName) {
    const el = document.getElementById("ia-thinking");
    if (!el) return;
    const label = TOOL_LABELS[toolName] || toolName;
    el.className = "ia-tool-status";
    el.innerHTML = '<div class="ia-tool-spinner"></div>' + label + "...";
    const msgs = document.getElementById("ia-widget-messages");
    if (msgs) msgs.scrollTop = msgs.scrollHeight;
  }

  function addWidgetSuggestions(suggestions) {
    if (!suggestions || !suggestions.length) return;
    const msgs = document.getElementById("ia-widget-messages");
    const bar = document.createElement("div");
    bar.className = "ia-suggestions-bar";
    suggestions.forEach(function (s) {
      const btn = document.createElement("button");
      btn.textContent = s;
      btn.onclick = function () {
        document
          .querySelectorAll(".ia-suggestions-bar")
          .forEach(function (b) { b.remove(); });
        window.__ia_send(s);
      };
      bar.appendChild(btn);
    });
    msgs.appendChild(bar);
    msgs.scrollTop = msgs.scrollHeight;
  }

  async function handleSend() {
    const input = document.getElementById("ia-widget-input");
    const sendBtn = document.getElementById("ia-widget-send");
    const msg = input.value.trim();
    if (!msg) return;

    input.value = "";
    input.style.height = "auto";
    sendBtn.disabled = true;

    // Remove old suggestion bars
    document
      .querySelectorAll(".ia-suggestions-bar")
      .forEach(function (b) { b.remove(); });

    addMsg("user", msg);
    addThinking();

    try {
      const res = await fetch(BASE_URL + "/api/v1/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: msg,
          conversation_id: conversationId,
        }),
      });

      if (!res.ok) {
        removeThinking();
        const err = await res.json().catch(function () { return {}; });
        addMsg("error", "Error: " + (err.detail || res.statusText));
        return;
      }

      var reader = res.body.getReader();
      var decoder = new TextDecoder();
      var accumulated = "";
      var msgDiv = null;
      var buffer = "";
      var renderTimer = null;
      var msgs = document.getElementById("ia-widget-messages");

      function renderNow() {
        if (msgDiv && accumulated) {
          msgDiv.innerHTML = renderMd(accumulated);
          msgs.scrollTop = msgs.scrollHeight;
        }
      }

      while (true) {
        var result = await reader.read();
        if (result.done) break;

        buffer += decoder.decode(result.value, { stream: true });
        var lines = buffer.split("\n");
        buffer = lines.pop();

        for (var i = 0; i < lines.length; i++) {
          var line = lines[i];
          if (line.indexOf("data: ") !== 0) continue;
          var event;
          try { event = JSON.parse(line.slice(6)); } catch (e) { continue; }

          if (event.type === "token") {
            if (!msgDiv) {
              removeThinking();
              var welcome = document.getElementById("ia-welcome-msg");
              if (welcome) welcome.style.display = "none";
              msgDiv = document.createElement("div");
              msgDiv.className = "ia-msg assistant";
              msgs.appendChild(msgDiv);
            }
            accumulated += event.content;
            if (!renderTimer) {
              renderTimer = setTimeout(function () {
                renderNow();
                renderTimer = null;
              }, 80);
            }

          } else if (event.type === "tool_start") {
            updateThinkingTool(event.name);

          } else if (event.type === "tool_result") {
            var thinkEl = document.getElementById("ia-thinking");
            if (thinkEl) {
              thinkEl.className = "ia-thinking";
              thinkEl.innerHTML =
                'Thinking<div class="dots"><span>.</span><span>.</span><span>.</span></div>';
            }

          } else if (event.type === "done") {
            removeThinking();
            conversationId = event.conversation_id;

            if (renderTimer) { clearTimeout(renderTimer); renderTimer = null; }
            if (msgDiv && accumulated) {
              renderNow();
            } else if (!msgDiv && accumulated) {
              msgDiv = addMsg("assistant", accumulated);
            }

            addWidgetSuggestions(event.suggestions);
          }
        }
      }

      if (renderTimer) { clearTimeout(renderTimer); renderTimer = null; }
      if (msgDiv && accumulated) renderNow();

    } catch (err) {
      removeThinking();
      addMsg("error", "Connection error: " + err.message);
    } finally {
      sendBtn.disabled = false;
      input.focus();
    }
  }

  // ---------- Init ----------
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", inject);
  } else {
    inject();
  }
})();
