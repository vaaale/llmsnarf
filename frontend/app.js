"use strict";

const API = "api";

/* ───────── helpers ───────── */
async function apiGet(path) {
  const res = await fetch(`${API}/${path}`);
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
  return res.json();
}

async function apiPut(path, body) {
  const res = await fetch(`${API}/${path}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    let detail = `${res.status}`;
    try { detail = (await res.json()).detail || detail; } catch (e) { /* ignore */ }
    throw new Error(detail);
  }
  return res.json();
}

function esc(s) {
  return String(s ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

function toast(msg, isError = false) {
  const t = document.getElementById("toast");
  t.textContent = msg;
  t.classList.toggle("error", isError);
  t.classList.add("show");
  setTimeout(() => t.classList.remove("show"), 2600);
}

function fmtTime(iso) {
  if (!iso) return "–";
  const idx = iso.indexOf("T");
  return idx >= 0 ? iso.slice(idx + 1, idx + 13) : iso;
}

function fmtDuration(ms) {
  if (ms === null || ms === undefined) return "–";
  if (ms < 1000) return Math.round(ms) + "ms";
  return (ms / 1000).toFixed(1) + "s";
}

function statusTag(t) {
  if (t.error) return '<span class="tag err" title="' + esc(t.error) + '">error</span>';
  if (t.status_code === null || t.status_code === undefined) return '<span class="tag warn">?</span>';
  return `<span class="tag ${t.status_code < 400 ? "ok" : "err"}">${t.status_code}</span>`;
}

function typeTag(t) {
  return `<span class="tag ${t.stream ? "stream" : "sync"}">${t.stream ? "stream" : "sync"}</span>`;
}

function jsonHtml(obj) {
  return esc(JSON.stringify(obj, null, 2))
    .replace(/&quot;([^&]+?)&quot;:/g, '<span class="j-key" style="color:var(--accent)">"$1"</span>:');
}

/* ───────── navigation ───────── */
const titles = { dashboard: "Dashboard", traces: "Traces", live: "Live Tail", config: "Configuration", ledger: "Validation Ledger" };
document.querySelectorAll(".nav-item").forEach((item) => {
  item.onclick = () => switchView(item.dataset.view);
});

function switchView(view) {
  document.querySelectorAll(".nav-item").forEach((i) => i.classList.toggle("active", i.dataset.view === view));
  document.querySelectorAll(".view").forEach((v) => v.classList.remove("active"));
  document.getElementById("view-" + view).classList.add("active");
  document.getElementById("view-title").textContent = titles[view];
  if (view === "dashboard") loadDashboard();
  if (view === "traces") loadTraces();
  if (view === "config") loadConfig();
  if (view === "ledger") loadLedger();
}

document.getElementById("refresh-btn").onclick = () => {
  const active = document.querySelector(".nav-item.active").dataset.view;
  switchView(active);
};

/* ───────── dashboard ───────── */
async function loadDashboard() {
  try {
    const [stats, recent] = await Promise.all([apiGet("stats?hours=24"), apiGet("traces?limit=10")]);

    document.getElementById("kpi-total").textContent = stats.total.toLocaleString();
    document.getElementById("kpi-errors").textContent = (stats.error_rate * 100).toFixed(1) + "%";
    document.getElementById("kpi-latency").innerHTML = stats.avg_duration_ms !== null
      ? `${Math.round(stats.avg_duration_ms)}<span class="kpi-unit"> ms</span>`
      : "–";
    const streamCount = recent.filter((t) => t.stream).length;
    document.getElementById("kpi-stream").textContent = recent.length
      ? Math.round((streamCount / recent.length) * 100) + "%"
      : "–";

    const chart = document.getElementById("traffic-chart");
    chart.innerHTML = "";
    const max = Math.max(...stats.requests_per_hour, 1);
    stats.requests_per_hour.forEach((v, i) => {
      const b = document.createElement("div");
      const errs = stats.errors_per_hour[i];
      b.className = "bar" + (errs > 0 && errs >= v / 2 ? " err" : "");
      b.style.height = (v / max) * 100 + "%";
      b.dataset.v = v + " req" + (errs ? ` · ${errs} errors` : "");
      chart.appendChild(b);
    });

    renderDist("model-dist", stats.by_model, stats.total, "var(--purple)");
    renderDist("key-dist", stats.by_api_key, stats.total, "var(--accent)");

    document.getElementById("recent-tbody").innerHTML = recent.map((t) => `
      <tr onclick="openTraceFromDashboard('${esc(t.id)}')">
        <td class="mono muted">${fmtTime(t.timestamp)}</td>
        <td><span class="tag model">${esc(t.model || "?")}</span></td>
        <td class="mono muted">${esc(t.endpoint)}</td>
        <td>${typeTag(t)}</td>
        <td>${statusTag(t)}</td>
        <td class="muted">${t.message_count}</td>
        <td class="mono muted">${fmtDuration(t.duration_ms)}</td>
      </tr>`).join("") || '<tr><td colspan="7" class="empty">No traces yet</td></tr>';
  } catch (e) {
    toast("Failed to load dashboard: " + e.message, true);
  }
}

function renderDist(elId, counts, total, color) {
  const rows = Object.entries(counts);
  document.getElementById(elId).innerHTML = rows.length
    ? rows.map(([name, n]) => {
        const pct = total ? Math.round((n / total) * 100) : 0;
        return `<div class="dist-row">
          <span class="mono" title="${esc(name)}">${esc(name)}</span>
          <div class="dist-bar-wrap"><div class="dist-bar" style="width:${pct}%;background:${color}"></div></div>
          <span class="dist-pct">${pct}%</span>
        </div>`;
      }).join("")
    : '<div class="muted" style="font-size:12px">No data</div>';
}

function openTraceFromDashboard(id) {
  switchView("traces");
  selectTrace(id);
}

/* ───────── traces ───────── */
let traceFilters = { q: "", model: "", api_key: "", status: "", correlation_id: "" };
let selectedTraceId = null;

function threadTag(t) {
  if (!t.correlation_id) return '<span class="muted" style="font-size:11px">–</span>';
  const short = t.correlation_id.length > 12 ? t.correlation_id.slice(0, 12) + "…" : t.correlation_id;
  return `<span class="tag route" title="${esc(t.correlation_id)}" onclick="filterByThread('${esc(t.correlation_id)}');event.stopPropagation();">🧵 ${esc(short)}</span>`;
}

function filterByThread(cid) {
  traceFilters.correlation_id = cid;
  const sel = document.getElementById("f-correlation");
  sel.value = cid;
  loadTraces();
}

async function loadTraces() {
  const params = new URLSearchParams({ limit: "200" });
  if (traceFilters.q) params.set("q", traceFilters.q);
  if (traceFilters.model) params.set("model", traceFilters.model);
  if (traceFilters.api_key) params.set("api_key", traceFilters.api_key);
  if (traceFilters.status) params.set("status", traceFilters.status);
  if (traceFilters.correlation_id) params.set("correlation_id", traceFilters.correlation_id);
  try {
    const traces = await apiGet("traces?" + params.toString());
    populateFilterOptions(traces);
    document.getElementById("traces-tbody").innerHTML = renderGroupedTraces(traces);
  } catch (e) {
    toast("Failed to load traces: " + e.message, true);
  }
}

function renderGroupedTraces(traces) {
  if (!traces.length) return '<tr><td colspan="7" class="empty">No traces match</td></tr>';

  const groups = [];
  const groupMap = new Map();
  for (const t of traces) {
    const cid = t.correlation_id || null;
    if (cid) {
      if (!groupMap.has(cid)) {
        const g = { cid, traces: [] };
        groupMap.set(cid, g);
        groups.push(g);
      }
      groupMap.get(cid).traces.push(t);
    } else {
      groups.push({ cid: null, traces: [t] });
    }
  }

  return groups.map((g) => {
    if (!g.cid) {
      const t = g.traces[0];
      return renderTraceRow(t, false);
    }
    const totalMsgs = g.traces.reduce((s, t) => s + (t.message_count || 0), 0);
    const firstTime = g.traces[0].timestamp;
    const short = g.cid.length > 16 ? g.cid.slice(0, 16) + "…" : g.cid;
    const safeCid = esc(g.cid);
    return `<tr class="thread-header" onclick="toggleThread('${safeCid}')">
        <td class="mono muted">${fmtTime(firstTime)}</td>
        <td colspan="5" style="font-weight:600">
          <span class="thread-toggle" id="toggle-${safeCid}">▼</span>
          <span class="tag route" title="${safeCid}">🧵 ${esc(short)}</span>
          <span class="muted" style="font-size:11px">${g.traces.length} requests · ${totalMsgs} msgs</span>
        </td>
        <td></td>
      </tr>` +
      g.traces.map((t) => {
        const row = renderTraceRow(t, true);
        return row.replace('id="row-', `data-thread="${safeCid}" id="row-`);
      }).join("");
  }).join("");
}

function renderTraceRow(t, nested) {
  const indent = nested ? 'padding-left:24px' : '';
  return `<tr id="row-${esc(t.id)}" class="${t.id === selectedTraceId ? "selected" : ""}${nested ? " thread-child" : ""}" onclick="selectTrace('${esc(t.id)}')">
        <td class="mono muted" style="${indent}">${fmtTime(t.timestamp)}</td>
        <td><span class="tag model">${esc(t.model || "?")}</span></td>
        <td>${typeTag(t)}</td>
        <td>${statusTag(t)}</td>
        <td class="muted">${t.message_count}</td>
        <td class="mono muted">${fmtDuration(t.duration_ms)}</td>
        <td>${threadTag(t)}</td>
      </tr>`;
}

function toggleThread(cid) {
  const toggle = document.getElementById("toggle-" + cid);
  const hidden = toggle && toggle.textContent === "▶";
  document.querySelectorAll(`tr[data-thread="${cid}"]`).forEach((r) => {
    r.style.display = hidden ? "" : "none";
  });
  if (toggle) toggle.textContent = hidden ? "▼" : "▶";
}

function populateFilterOptions(traces) {
  const fill = (id, values, current) => {
    const sel = document.getElementById(id);
    const first = sel.options[0].outerHTML;
    sel.innerHTML = first + [...new Set(values)].sort().map((v) =>
      `<option value="${esc(v)}" ${v === current ? "selected" : ""}>${esc(v)}</option>`).join("");
  };
  fill("f-model", traces.map((t) => t.model).filter(Boolean), traceFilters.model);
  fill("f-apikey", traces.map((t) => t.api_key).filter(Boolean), traceFilters.api_key);
  fill("f-correlation", traces.map((t) => t.correlation_id).filter(Boolean), traceFilters.correlation_id);
}

let searchTimer = null;
document.getElementById("trace-search").oninput = (e) => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => { traceFilters.q = e.target.value.trim(); loadTraces(); }, 300);
};
document.getElementById("f-model").onchange = (e) => { traceFilters.model = e.target.value; loadTraces(); };
document.getElementById("f-apikey").onchange = (e) => { traceFilters.api_key = e.target.value; loadTraces(); };
document.getElementById("f-status").onchange = (e) => { traceFilters.status = e.target.value; loadTraces(); };
document.getElementById("f-correlation").onchange = (e) => { traceFilters.correlation_id = e.target.value; loadTraces(); };
document.getElementById("trace-refresh").onclick = loadTraces;

async function selectTrace(id) {
  selectedTraceId = id;
  document.querySelectorAll("#traces-tbody tr").forEach((r) => r.classList.remove("selected"));
  const row = document.getElementById("row-" + id);
  if (row) row.classList.add("selected");

  let d;
  try {
    d = await apiGet("traces/" + encodeURIComponent(id));
  } catch (e) {
    toast("Failed to load trace: " + e.message, true);
    return;
  }
  const s = d.summary;

  document.getElementById("d-title").textContent = (s.model || "?") + " · " + fmtTime(s.timestamp);
  document.getElementById("d-file").textContent = s.id;
  document.getElementById("d-meta").innerHTML = `
    ${typeTag(s)} ${statusTag(s)}
    <span class="tag sync">key: ${esc(s.api_key)}</span>
    <span class="tag sync">${fmtDuration(s.duration_ms)}</span>
    <span class="tag route">${esc(s.endpoint)}</span>
    ${s.correlation_id ? `<span class="tag route" title="${esc(s.correlation_id)}">🧵 ${esc(s.correlation_id.length > 16 ? s.correlation_id.slice(0,16) + "…" : s.correlation_id)}</span>` : ""}`;

  renderConversation(d);
  document.getElementById("pane-req").innerHTML =
    `<pre class="json">${jsonHtml(d.request_payload)}</pre>`;
  document.getElementById("pane-resp").innerHTML = d.response_body !== null
    ? `<pre class="json">${jsonHtml(d.response_body)}</pre>`
    : d.assembled_content
      ? `<div class="assembled-note">⚡ Streaming response — assembled from ${d.response_chunks.length} chunks</div><pre class="json">${esc(d.assembled_content)}</pre>`
      : '<div class="empty">No response captured.</div>';
  document.getElementById("pane-chunks").innerHTML = d.response_chunks.length
    ? d.response_chunks.map((c, i) =>
        `<div class="chunk-row"><span class="chunk-t">#${i}</span><span class="chunk-delta">${esc(c.length > 400 ? c.slice(0, 400) + "…" : c)}</span></div>`).join("")
    : '<div class="empty">Non-streaming request — no SSE chunks.</div>';
  document.getElementById("pane-headers").innerHTML =
    `<h3 style="font-size:12px;color:var(--muted);margin-bottom:8px">REQUEST HEADERS</h3><pre class="json">${jsonHtml(d.request_headers)}</pre>` +
    `<h3 style="font-size:12px;color:var(--muted);margin:14px 0 8px">RESPONSE HEADERS</h3><pre class="json">${jsonHtml(d.response_headers)}</pre>`;
}

function messageContentToText(content) {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content.map((part) => (part && part.type === "text" ? part.text : `[${part?.type || "content"}]`)).join("\n");
  }
  return JSON.stringify(content, null, 2);
}

function renderConversation(d) {
  const payload = d.request_payload;
  const messages = extractRequestMessages(payload);
  let html = "";
  if (d.assembled_content) {
    html += `<div class="assembled-note">⚡ Streaming response — assistant message assembled from ${d.response_chunks.length} SSE chunks</div>`;
  }
  html += messages.map((m, i) => {
    const role = esc(m.role || m.type || "unknown");
    return `<div class="msg ${role}">
      <div class="msg-head" onclick="toggleMessage(${i})">
        <span class="msg-role">${role}</span>
        <span class="msg-toggle" id="msg-toggle-${i}">▼</span>
      </div>
      <div class="msg-body" id="msg-body-${i}">${esc(m.content)}</div>
    </div>`;
  }).join("");

  const responseParts = extractResponseParts(d);
  for (const part of responseParts) {
    const idx = messages.length + responseParts.indexOf(part);
    const role = esc(part.role || part.type || "assistant");
    html += `<div class="msg ${role}">
      <div class="msg-head" onclick="toggleMessage(${idx})">
        <span class="msg-role">${role}${part.note ? ` <span class="subst-note">${part.note}</span>` : ""}</span>
        <span class="msg-toggle" id="msg-toggle-${idx}">▼</span>
      </div>
      <div class="msg-body" id="msg-body-${idx}">${esc(part.content)}</div>
    </div>`;
  }

  if (d.summary.error) {
    const errorIndex = messages.length + responseParts.length + 1;
    html += `<div class="msg error">
      <div class="msg-head" onclick="toggleMessage(${errorIndex})">
        <span class="msg-role">error</span>
        <span class="msg-toggle" id="msg-toggle-${errorIndex}">▼</span>
      </div>
      <div class="msg-body" id="msg-body-${errorIndex}">${esc(d.summary.error)}</div>
    </div>`;
  }
  document.getElementById("pane-conv").innerHTML = html || '<div class="empty">No messages in payload.</div>';
}

function extractRequestMessages(payload) {
  if (!payload) return [];
  if (Array.isArray(payload.messages)) return payload.messages.map(normalizeMessage);
  if (Array.isArray(payload.input)) return payload.input.map(normalizeInputItem);
  if (typeof payload.input === "string") return [{ role: "user", content: payload.input }];
  return [];
}

function normalizeMessage(m) {
  return { role: m.role, content: messageContentToText(m.content) };
}

function normalizeInputItem(item) {
  if (item.role) {
    return { role: item.role, content: messageContentToText(item.content) };
  }
  if (item.type === "function_call") {
    return { type: "function_call", content: `call_id: ${item.call_id || ""}\nname: ${item.name || ""}\narguments: ${item.arguments || ""}` };
  }
  if (item.type === "function_call_output") {
    return { type: "function_call_output", content: `call_id: ${item.call_id || ""}\noutput: ${item.output || ""}` };
  }
  return { type: item.type || "unknown", content: JSON.stringify(item, null, 2) };
}

function extractResponseParts(d) {
  const parts = [];
  let assistant = d.assembled_content;
  if (!assistant && d.response_body) {
    if (Array.isArray(d.response_body.choices)) {
      const choice = d.response_body.choices[0];
      assistant = choice?.message?.content || choice?.text || "";
    } else if (Array.isArray(d.response_body.output)) {
      for (const item of d.response_body.output) {
        if (item.type === "message" && Array.isArray(item.content)) {
          const text = item.content
            .filter((c) => c.type === "output_text" || c.type === "text")
            .map((c) => c.text)
            .join("\n");
          if (text) parts.push({ role: item.role || "assistant", content: text, note: "response" });
        } else if (item.type === "reasoning") {
          const text = Array.isArray(item.summary)
            ? item.summary.map((s) => s.text || "").join("\n")
            : (item.summary || "");
          if (text) parts.push({ type: "reasoning", content: text, note: "thinking" });
        } else if (item.type === "function_call") {
          parts.push({ type: "function_call", content: `call_id: ${item.call_id || ""}\nname: ${item.name || ""}\narguments: ${item.arguments || ""}` });
        } else if (item.type === "function_call_output") {
          parts.push({ type: "function_call_output", content: `call_id: ${item.call_id || ""}\noutput: ${item.output || ""}` });
        } else if (item.type === "web_search_call") {
          const query = item.action?.query || "";
          parts.push({ type: "web_search", content: `query: ${query}`, note: "search" });
        }
      }
      assistant = null;
    }
  }
  if (assistant) parts.push({ role: "assistant", content: assistant, note: "response" });
  return parts;
}

function toggleMessage(index) {
  const body = document.getElementById(`msg-body-${index}`);
  const toggle = document.getElementById(`msg-toggle-${index}`);
  
  if (body.style.display === 'none') {
    body.style.display = 'block';
    toggle.textContent = '▼';
    toggle.style.transform = 'rotate(0deg)';
  } else {
    body.style.display = 'none';
    toggle.textContent = '▶';
    toggle.style.transform = 'rotate(0deg)';
  }
}

document.querySelectorAll("#d-tabs .tab").forEach((tab) => {
  tab.onclick = () => {
    document.querySelectorAll("#d-tabs .tab").forEach((x) => x.classList.remove("active"));
    document.querySelectorAll(".tab-pane").forEach((x) => x.classList.remove("active"));
    tab.classList.add("active");
    document.getElementById("pane-" + tab.dataset.tab).classList.add("active");
  };
});

/* ───────── live tail ───────── */
let tailOn = true;
let tailSeen = new Set();
let tailCount = 0;

function tailThreadKey(cid) { return cid ? "tail-thread-" + cid : null; }

function ensureTailThread(box, cid) {
  const key = tailThreadKey(cid);
  if (!key) return box;
  let el = document.getElementById(key);
  if (el) return el;
  el = document.createElement("div");
  el.className = "tail-thread";
  el.id = key;
  const short = cid.length > 16 ? cid.slice(0, 16) + "…" : cid;
  el.innerHTML = `<div class="tail-thread-header" onclick="this.parentElement.classList.toggle('collapsed')">
    <span class="thread-toggle">▼</span>
    <span class="tag route" title="${esc(cid)}">🧵 ${esc(short)}</span>
  </div><div class="tail-thread-body"></div>`;
  box.prepend(el);
  return el.querySelector(".tail-thread-body");
}

async function pollTail() {
  if (!tailOn || !document.getElementById("view-live").classList.contains("active")) return;
  try {
    const traces = await apiGet("traces?limit=25");
    const box = document.getElementById("tail-box");
    for (const t of traces.reverse()) {
      if (tailSeen.has(t.id)) continue;
      tailSeen.add(t.id);
      const container = ensureTailThread(box, t.correlation_id);
      const line = document.createElement("div");
      line.className = "tail-line new" + (t.correlation_id ? " tail-nested" : "");
      line.onclick = () => openTraceFromDashboard(t.id);
      line.innerHTML = `<span class="t">${fmtTime(t.timestamp)}</span>
        ${statusTag(t)} <span class="tag model">${esc(t.model || "?")}</span>
        <span class="muted">${esc(t.endpoint)} · ${fmtDuration(t.duration_ms)} · ${t.message_count} msgs · ${esc(t.api_key)}</span>
        ${t.correlation_id ? `<span class="tag route" title="${esc(t.correlation_id)}">🧵 ${esc(t.correlation_id.length > 12 ? t.correlation_id.slice(0,12) + "…" : t.correlation_id)}</span>` : ""}`;
      container.prepend(line);
      tailCount++;
    }
    while (box.children.length > 100) box.lastChild.remove();
    document.getElementById("tail-count").textContent = tailCount + " requests observed";
  } catch (e) { /* transient poll errors are ignored */ }
}
setInterval(pollTail, 3000);

document.getElementById("tail-toggle").onclick = (e) => {
  tailOn = !tailOn;
  e.target.textContent = tailOn ? "● Following" : "⏸ Paused";
  e.target.classList.toggle("live", tailOn);
};

/* ───────── configuration ───────── */
let CONFIG = null;
let editingIndex = null;
let modelChips = [];

async function loadConfig() {
  try {
    CONFIG = await apiGet("config");
    document.getElementById("cfg-listen").textContent = `${CONFIG.host || "0.0.0.0"}:${CONFIG.port || 8000}`;
    document.getElementById("cfg-logs").textContent = CONFIG.logs_dir || "./logs";
    document.getElementById("cfg-traces").textContent = CONFIG.trace_dir || "./logs";
    document.getElementById("hdr-trace-dir").textContent = CONFIG.trace_dir || "./logs";
    document.getElementById("footer-info").textContent = `${CONFIG.host || "0.0.0.0"}:${CONFIG.port || 8000}`;
    renderConfigCards();
    renderWebSearch();
    renderWebFetch();
  } catch (e) {
    toast("Failed to load config: " + e.message, true);
  }
}

function renderWebSearch() {
  const ws = CONFIG.web_search || {};
  document.getElementById("ws-enabled").classList.toggle("off", !ws.enabled);
  document.getElementById("ws-url").value = ws.base_url || "";
  document.getElementById("ws-format").value = ws.format || "markdown";
  document.getElementById("ws-extract").value = ws.extract ?? 3;
  document.getElementById("ws-extract-mode").value = ws.extract_mode || "auto";
  document.getElementById("ws-limit").value = ws.limit ?? 25;
  document.getElementById("ws-filter").classList.toggle("off", !ws.filter);
  document.getElementById("ws-mode").value = ws.mode || "balanced";
  const activeEngines = new Set(ws.engines || []);
  document.querySelectorAll("#ws-engines-group input[type=checkbox]").forEach((cb) => {
    cb.checked = activeEngines.has(cb.value);
  });
  const base = ws.base_url || "…";
  document.getElementById("ws-hint-url").textContent = base + "/mega/search";
}

function renderWebFetch() {
  const wf = CONFIG.web_fetch || {};
  document.getElementById("wf-url").value = wf.base_url || "";
  document.getElementById("wf-format").value = wf.format || "markdown";
  document.getElementById("wf-mode").value = wf.mode || "auto";
  document.getElementById("wf-hint-url").textContent = (wf.base_url || "…") + "/extract";
}

document.getElementById("ws-enabled").onclick = function () { this.classList.toggle("off"); };
document.getElementById("ws-filter").onclick = function () { this.classList.toggle("off"); };

document.getElementById("ws-save-btn").onclick = async () => {
  const baseUrl = document.getElementById("ws-url").value.trim();
  if (!baseUrl) { toast("OpenSERP base URL is required", true); return; }
  const extract = parseInt(document.getElementById("ws-extract").value, 10);
  const limit = parseInt(document.getElementById("ws-limit").value, 10);
  if (isNaN(extract) || extract < 0 || extract > 5) { toast("Extract must be between 0 and 5", true); return; }
  if (isNaN(limit) || limit < 1 || limit > 100) { toast("Limit must be between 1 and 100", true); return; }
  const engines = Array.from(document.querySelectorAll("#ws-engines-group input[type=checkbox]"))
    .filter((cb) => cb.checked).map((cb) => cb.value);
  const payload = {
    enabled: !document.getElementById("ws-enabled").classList.contains("off"),
    base_url: baseUrl,
    format: document.getElementById("ws-format").value,
    extract,
    extract_mode: document.getElementById("ws-extract-mode").value,
    limit,
    filter: !document.getElementById("ws-filter").classList.contains("off"),
    mode: document.getElementById("ws-mode").value,
    engines,
  };
  try {
    CONFIG = await apiPut("config/web_search", payload);
    renderWebSearch();
    renderWebFetch();
    toast("Saved llmproxy.yaml — web search settings active");
  } catch (e) {
    toast("Save failed: " + e.message, true);
  }
};

document.getElementById("wf-save-btn").onclick = async () => {
  const baseUrl = document.getElementById("wf-url").value.trim();
  if (!baseUrl) { toast("Web fetch base URL is required", true); return; }
  const payload = {
    base_url: baseUrl,
    format: document.getElementById("wf-format").value,
    mode: document.getElementById("wf-mode").value,
  };
  try {
    CONFIG = await apiPut("config/web_fetch", payload);
    renderWebFetch();
    toast("Saved llmproxy.yaml — web fetch settings active");
  } catch (e) {
    toast("Save failed: " + e.message, true);
  }
};

function mapHtml(obj) {
  const entries = Object.entries(obj || {});
  if (!entries.length) return '<span class="muted">—</span>';
  return entries.map(([a, b]) => `${esc(a)}<span class="alias-arrow">→</span>${esc(b)}`).join("<br>");
}

function renderConfigCards() {
  document.getElementById("config-grid").innerHTML = CONFIG.endpoints.map((ep, i) => `
    <div class="ep-card" style="${ep.enabled ? "" : "opacity:.55"}">
      <div class="ep-head">
        <span class="name">${esc(ep.name)}</span>
        ${ep.models.includes("*") ? '<span class="tag route">wildcard *</span>' : ""}
        ${ep.enabled ? "" : '<span class="tag warn">disabled</span>'}
        <div class="ep-actions">
          <div class="toggle ${ep.log ? "" : "off"}" title="Trace logging" onclick="quickToggleLog(${i})"></div>
          <button class="icon-btn" title="Edit" onclick="openEditor(${i})">✎</button>
          <button class="icon-btn danger" title="Delete" onclick="deleteEndpoint(${i})">🗑</button>
        </div>
      </div>
      <div class="ep-body">
        <div class="kv"><span class="k">Base URL</span><span class="v">${esc(ep.base_url)}</span></div>
        <div class="kv"><span class="k">API key</span><span class="v">${esc(ep.api_key_masked) || "—"}</span></div>
        <div class="kv"><span class="k">Models</span><span class="v">${ep.models.map((m) => '<span class="tag model">' + esc(m) + "</span>").join(" ")}</span></div>
        <div class="kv"><span class="k">Aliases</span><span class="v">${mapHtml(ep.aliases)}</span></div>
        <div class="kv"><span class="k">Role subst.</span><span class="v">${mapHtml(ep.substitute_role)}</span></div>
        <div class="kv"><span class="k">Max models</span><span class="v">${ep.max_models > 0 ? esc(ep.max_models) : '<span class="muted">unlimited</span>'}</span></div>
        <div class="kv"><span class="k">Trace log</span><span class="v"><span class="tag ${ep.log ? "ok" : "warn"}">${ep.log ? "enabled" : "off"}</span></span></div>
      </div>
    </div>`).join("") || '<div class="empty">No endpoints configured.</div>';
}

async function persistConfig() {
  const payload = {
    endpoints: CONFIG.endpoints.map((ep) => ({
      name: ep.name,
      base_url: ep.base_url,
      api_key: ep.api_key || null,
      models: ep.models,
      aliases: ep.aliases || {},
      substitute_role: ep.substitute_role || {},
      log: ep.log,
      enabled: ep.enabled,
      max_models: ep.max_models || 0,
    })),
  };
  CONFIG = await apiPut("config", payload);
  renderConfigCards();
}

async function quickToggleLog(i) {
  CONFIG.endpoints[i].log = !CONFIG.endpoints[i].log;
  try {
    await persistConfig();
    toast("Saved llmproxy.yaml");
  } catch (e) {
    toast("Save failed: " + e.message, true);
    loadConfig();
  }
}

async function deleteEndpoint(i) {
  if (!confirm(`Delete endpoint "${CONFIG.endpoints[i].name}"?`)) return;
  CONFIG.endpoints.splice(i, 1);
  try {
    await persistConfig();
    toast("Endpoint deleted — llmproxy.yaml updated");
  } catch (e) {
    toast("Save failed: " + e.message, true);
    loadConfig();
  }
}

document.getElementById("cfg-reload-btn").onclick = () => { loadConfig(); toast("Config reloaded"); };
document.getElementById("add-ep-btn").onclick = () => openEditor(null);

/* modal */
function addMapRow(containerId, k = "", v = "") {
  const row = document.createElement("div");
  row.className = "map-row";
  row.innerHTML = `<input type="text" placeholder="from" value="${esc(k)}">
    <span class="arrow">→</span>
    <input type="text" placeholder="to" value="${esc(v)}">
    <button class="icon-btn danger">✕</button>`;
  row.querySelector("button").onclick = () => row.remove();
  document.getElementById(containerId).appendChild(row);
}

document.querySelectorAll(".add-row-btn").forEach((btn) => {
  btn.onclick = () => addMapRow(btn.dataset.target);
});

function renderChips() {
  const box = document.getElementById("fe-models");
  box.querySelectorAll(".chip").forEach((c) => c.remove());
  const input = document.getElementById("fe-models-input");
  modelChips.forEach((m, i) => {
    const chip = document.createElement("div");
    chip.className = "chip";
    chip.innerHTML = `${esc(m)}<span>✕</span>`;
    chip.querySelector("span").onclick = () => { modelChips.splice(i, 1); renderChips(); };
    box.insertBefore(chip, input);
  });
}

document.getElementById("fe-models").onclick = function () { this.querySelector("input").focus(); };
document.getElementById("fe-models-input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && e.target.value.trim()) {
    e.preventDefault();
    const v = e.target.value.trim();
    if (!modelChips.includes(v)) modelChips.push(v);
    e.target.value = "";
    renderChips();
  } else if (e.key === "Backspace" && !e.target.value && modelChips.length) {
    modelChips.pop();
    renderChips();
  }
});

function openEditor(i) {
  editingIndex = i;
  const ep = i === null
    ? { name: "", base_url: "", api_key_masked: "", models: [], aliases: {}, substitute_role: {}, log: true, enabled: true, max_models: 0 }
    : CONFIG.endpoints[i];
  document.getElementById("modal-title").textContent = i === null ? "Add endpoint" : "Edit endpoint — " + ep.name;
  document.getElementById("fe-name").value = ep.name;
  document.getElementById("fe-url").value = ep.base_url;
  document.getElementById("fe-key").value = "";
  document.getElementById("fe-key").placeholder = ep.api_key_masked || "sk-…";
  document.getElementById("fe-key-hint").textContent = i === null
    ? "Optional. Leave empty to forward the client's Authorization header."
    : `Current: ${ep.api_key_masked || "not set"}. Leave empty to keep it.`;
  document.getElementById("fe-enabled").classList.toggle("off", !ep.enabled);
  document.getElementById("fe-log").classList.toggle("off", !ep.log);
  document.getElementById("fe-max-models").value = ep.max_models || 0;
  modelChips = [...ep.models];
  renderChips();
  document.getElementById("fe-aliases").innerHTML = "";
  document.getElementById("fe-roles").innerHTML = "";
  Object.entries(ep.aliases || {}).forEach(([k, v]) => addMapRow("fe-aliases", k, v));
  Object.entries(ep.substitute_role || {}).forEach(([k, v]) => addMapRow("fe-roles", k, v));
  document.getElementById("ep-modal").classList.add("open");
}

function closeEditor() {
  document.getElementById("ep-modal").classList.remove("open");
}

document.getElementById("modal-close").onclick = closeEditor;
document.getElementById("modal-cancel").onclick = closeEditor;
document.getElementById("ep-modal").addEventListener("click", (e) => {
  if (e.target.id === "ep-modal") closeEditor();
});
document.getElementById("fe-enabled").onclick = function () { this.classList.toggle("off"); };
document.getElementById("fe-log").onclick = function () { this.classList.toggle("off"); };

function collectMap(containerId) {
  const out = {};
  document.querySelectorAll(`#${containerId} .map-row`).forEach((row) => {
    const [k, v] = row.querySelectorAll("input");
    if (k.value.trim() && v.value.trim()) out[k.value.trim()] = v.value.trim();
  });
  return out;
}

document.getElementById("modal-apply").onclick = async () => {
  const name = document.getElementById("fe-name").value.trim();
  const url = document.getElementById("fe-url").value.trim();
  if (!name || !url) { toast("Endpoint name and Base URL are required", true); return; }
  if (!modelChips.length) { toast("At least one model (or *) is required", true); return; }
  const clash = CONFIG.endpoints.some((ep, i) => ep.name === name && i !== editingIndex);
  if (clash) { toast(`An endpoint named "${name}" already exists`, true); return; }

  const newKey = document.getElementById("fe-key").value.trim();
  const ep = {
    name,
    base_url: url,
    api_key: newKey || null,
    api_key_masked: "",
    models: [...modelChips],
    aliases: collectMap("fe-aliases"),
    substitute_role: collectMap("fe-roles"),
    log: !document.getElementById("fe-log").classList.contains("off"),
    enabled: !document.getElementById("fe-enabled").classList.contains("off"),
    max_models: parseInt(document.getElementById("fe-max-models").value, 10) || 0,
  };
  if (editingIndex === null) CONFIG.endpoints.push(ep);
  else CONFIG.endpoints[editingIndex] = ep;

  try {
    await persistConfig();
    closeEditor();
    toast("Saved llmproxy.yaml — active immediately");
  } catch (e) {
    toast("Save failed: " + e.message, true);
    loadConfig();
  }
};

/* routing tester */
document.getElementById("route-btn").onclick = async () => {
  const model = document.getElementById("route-input").value.trim();
  if (!model) return;
  const el = document.getElementById("route-result");
  el.style.display = "block";
  try {
    const r = await apiGet("config/resolve?model=" + encodeURIComponent(model));
    if (r.endpoint) {
      el.style.borderColor = r.wildcard ? "rgba(79,156,249,.3)" : "rgba(63,185,80,.3)";
      el.style.background = r.wildcard ? "rgba(79,156,249,.08)" : "rgba(63,185,80,.08)";
      el.innerHTML = `<b>${esc(model)}</b> → ${r.wildcard ? "wildcard " : ""}endpoint <span class="tag route">${esc(r.endpoint)}</span>
        · forwarded as <code>${esc(r.forwarded_model)}</code>${r.forwarded_model !== model ? " (aliased)" : ""}
        · target <code class="mono">${esc(r.base_url)}</code>
        · tracing <span class="tag ${r.log ? "ok" : "warn"}">${r.log ? "on" : "off"}</span>`;
    } else {
      el.style.borderColor = "rgba(248,81,73,.3)";
      el.style.background = "rgba(248,81,73,.08)";
      el.innerHTML = `<b>${esc(model)}</b> → <span class="tag err">404 routing_error</span> · no endpoint matches and no wildcard fallback configured`;
    }
  } catch (e) {
    toast("Resolve failed: " + e.message, true);
  }
};

/* ───────── validation ledger ───────── */
const ISSUE_CODE_LABELS = {
  "response.unclosed_tag": { label: "Unclosed tag", cls: "warn" },
  "response.tool_call.invalid_json": { label: "Invalid JSON args", cls: "err" },
  "response.tool_call.args_not_object": { label: "Args not object", cls: "err" },
  "response.tool_call.null_arg": { label: "Null arg", cls: "warn" },
  "response.tool_call.autolink_arg": { label: "Autolink arg", cls: "warn" },
  "response.tool_call.stringified_array": { label: "Stringified array", cls: "warn" },
  "response.truncated": { label: "Truncated", cls: "warn" },
  "request.missing_model": { label: "No model", cls: "err" },
  "request.not_object": { label: "Bad request", cls: "err" },
  "request.messages_not_array": { label: "Bad messages", cls: "err" },
  "request.message_not_object": { label: "Bad message", cls: "err" },
  "request.message_missing_role": { label: "Missing role", cls: "warn" },
};

function issueTag(code) {
  const meta = ISSUE_CODE_LABELS[code] || { label: code, cls: "warn" };
  return `<span class="tag ${meta.cls}" title="${esc(code)}">${esc(meta.label)}</span>`;
}

let ledgerEntries = [];
let selectedLedgerEntryId = null;

async function loadLedger() {
  const modelFilter = document.getElementById("ledger-model-filter").value.trim();
  const params = new URLSearchParams({ limit: "200" });
  if (modelFilter) params.set("model", modelFilter);
  try {
    ledgerEntries = await apiGet("ledger?" + params.toString());
    const count = ledgerEntries.length;
    document.getElementById("ledger-count").textContent = count
      ? count + " entr" + (count === 1 ? "y" : "ies") + " with issues"
      : "No validation issues recorded yet";
    document.getElementById("ledger-tbody").innerHTML = ledgerEntries.map((e) => {
      const codes = [...new Set(e.issues.map((i) => i.code))];
      return `<tr id="lrow-${esc(e.id)}" class="${e.id === selectedLedgerEntryId ? "selected" : ""}" onclick="selectLedgerEntry('${esc(e.id)}')">
        <td class="mono muted">${fmtTime(e.timestamp)}</td>
        <td><span class="tag model">${esc(e.model || "?")}</span></td>
        <td class="mono muted">${esc(e.endpoint)}</td>
        <td>${codes.map(issueTag).join(" ")}</td>
      </tr>`;
    }).join("") || '<tr><td colspan="4" class="empty">No entries</td></tr>';
    if (selectedLedgerEntryId) {
      const still = ledgerEntries.find((e) => e.id === selectedLedgerEntryId);
      if (still) renderLedgerDetail(still);
    }
  } catch (ex) {
    toast("Failed to load ledger: " + ex.message, true);
  }
}

function selectLedgerEntry(id) {
  selectedLedgerEntryId = id;
  document.querySelectorAll("#ledger-tbody tr").forEach((r) => r.classList.remove("selected"));
  const row = document.getElementById("lrow-" + id);
  if (row) row.classList.add("selected");
  const entry = ledgerEntries.find((e) => e.id === id);
  if (entry) renderLedgerDetail(entry);
}

function renderLedgerDetail(entry) {
  document.getElementById("ledger-d-title").textContent =
    (entry.model || "?") + " · " + fmtTime(entry.timestamp);
  document.getElementById("ledger-d-meta").innerHTML =
    `<span class="tag model">${esc(entry.model || "?")}</span>` +
    `<span class="tag route">${esc(entry.endpoint)}</span>` +
    `<span class="tag sync" title="trace id">${esc(entry.trace_id)}</span>` +
    `<button class="btn" style="padding:3px 10px;font-size:11px" onclick="openTraceFromLedger('${esc(entry.trace_id)}')">↗ Open trace</button>`;

  const grouped = {};
  for (const issue of entry.issues) {
    (grouped[issue.code] = grouped[issue.code] || []).push(issue);
  }
  const html = Object.entries(grouped).map(([code, issues]) => {
    const meta = ISSUE_CODE_LABELS[code] || { label: code, cls: "warn" };
    const rows = issues.map((i) =>
      `<div style="padding:6px 0;border-bottom:1px solid rgba(43,52,68,.4);font-size:12px;font-family:monospace">
        ${i.path ? `<span style="color:var(--muted)">${esc(i.path)}</span><br>` : ""}
        <span style="color:var(--text)">${esc(i.message)}</span>
      </div>`
    ).join("");
    return `<div style="margin-bottom:16px">
      <div style="font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin-bottom:6px">
        ${issueTag(code)} <span style="margin-left:6px">${esc(meta.label)}</span>
      </div>
      ${rows}
    </div>`;
  }).join("");

  document.getElementById("ledger-d-body").innerHTML = html ||
    '<div class="empty">No issues.</div>';
}

function openTraceFromLedger(traceId) {
  switchView("traces");
  selectTrace(traceId);
}

let ledgerSearchTimer = null;
document.getElementById("ledger-model-filter").oninput = () => {
  clearTimeout(ledgerSearchTimer);
  ledgerSearchTimer = setTimeout(loadLedger, 300);
};
document.getElementById("ledger-refresh").onclick = loadLedger;

/* ───────── init ───────── */
loadConfig();
loadDashboard();
