/* ================================================================
   Comment Intelligence — App JS (v3 — industry-standard rewrite)
   ================================================================ */

let sid = sessionStorage.getItem('sid');
if (!sid) { sid = crypto.randomUUID(); sessionStorage.setItem('sid', sid); }

const S = { sid, count: 0, meta: null, plat: null, analysis: null, rag: false };
let chartCounter = 0; // avoid Date.now() ID collisions
let ragPollTimer = null;

// ─── Helpers ────────────────────────────────────────────────────
async function api(url, opts = {}) {
  const r = await fetch(url, {
    method: opts.method || 'GET',
    headers: { 'Content-Type': 'application/json' },
    body: opts.body ? JSON.stringify(opts.body) : undefined
  });
  if (!r.ok) {
    const e = await r.json().catch(() => ({ detail: r.statusText }));
    throw new Error(e.detail || 'Request failed');
  }
  return r.json();
}

function toast(msg, type = 'ok') {
  const c = document.getElementById('toasts');
  const d = document.createElement('div');
  d.className = 'toast toast-' + type;
  d.textContent = msg;
  c.appendChild(d);
  setTimeout(() => { d.style.opacity = '0'; setTimeout(() => d.remove(), 300); }, 4000);
}

function esc(s) { if (!s) return ''; const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }
function fmt(n) { return new Intl.NumberFormat().format(n || 0); }

function uid(prefix) { return prefix + '_' + (++chartCounter); }

// ─── Init ───────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {

  // Try recover session
  api(`/api/session/${S.sid}`).then(r => {
    if (r.count > 0) {
      S.count = r.count; S.meta = r.metadata; S.plat = r.platform;
      updateBadge();
      showExportBtn();
      if (r.has_analysis) {
        S.analysis = r.analysis_results;
        render(r.analysis_results, []);
      }
      if (r.rag_ready) {
        S.rag = true;
      } else if (r.rag_status === 'building') {
        startRagPolling();
      }
    }
  }).catch(() => {});

  // Buttons
  $('fetch-btn').addEventListener('click', run);
  $('url-input').addEventListener('keydown', e => { if (e.key === 'Enter') run(); });
  $('new-session-btn').addEventListener('click', () => { sessionStorage.removeItem('sid'); location.reload(); });
  $('send-btn').addEventListener('click', send);
  $('chat-input').addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) send(); });
  $('clear-chat-btn').addEventListener('click', clearChat);
  $('export-btn').addEventListener('click', exportCSV);
});

function $(id) { return document.getElementById(id); }

function updateBadge() {
  const p = (S.plat||'').charAt(0).toUpperCase() + (S.plat||'').slice(1);
  $('status-indicator').innerHTML =
    `<span class="badge badge-green">${fmt(S.count)} comments</span>` +
    `<span class="badge badge-gray">${p}</span>`;
}

function showExportBtn() {
  $('export-btn').style.display = 'block';
}

// ─── Pipeline ───────────────────────────────────────────────────
async function run() {
  const url = $('url-input').value.trim();
  if (!url) return toast('Paste a URL first', 'err');

  const maxComments = parseInt($('comment-count').value, 10) || 500;
  const btn = $('fetch-btn');
  btn.disabled = true; btn.textContent = 'Working...';

  // Show progress + loading
  $('progress-wrap').classList.remove('hidden');
  $('progress-bar').style.width = '5%';
  $('progress-label').textContent = 'Connecting...';
  $('welcome').classList.add('hidden');
  $('results').classList.add('hidden');
  $('loading').classList.remove('hidden');
  $('loading-msg').textContent = 'Fetching comments...';
  $('loading-sub').textContent = `Requesting up to ${fmt(maxComments)} comments...`;

  // Progress stages — smoother with target-based interpolation
  let currentPct = 5;
  let targetPct = 25;
  const tick = setInterval(() => {
    if (currentPct < targetPct) {
      currentPct = Math.min(currentPct + 1.5, targetPct);
      $('progress-bar').style.width = currentPct + '%';
    }
  }, 300);

  // Stage transitions
  setTimeout(() => {
    targetPct = 40;
    $('progress-label').textContent = 'Analyzing with Gemini...';
    $('loading-msg').textContent = 'Analyzing with Gemini...';
    $('loading-sub').textContent = 'Running sentiment analysis, topic discovery, and summary generation...';
  }, 3000);

  setTimeout(() => {
    targetPct = 70;
  }, 10000);

  setTimeout(() => {
    targetPct = 85;
    $('progress-label').textContent = 'Almost done...';
    $('loading-msg').textContent = 'Finalizing analysis...';
  }, 20000);

  try {
    const r = await api('/api/fetch', { method: 'POST', body: { url, max_comments: maxComments, session_id: S.sid } });
    clearInterval(tick);
    $('progress-bar').style.width = '100%';
    $('progress-label').textContent = 'Done!';

    S.count = r.count; S.meta = r.metadata; S.plat = r.platform; S.analysis = r.analysis;
    S.rag = r.rag_ready;
    updateBadge();
    showExportBtn();

    $('loading').classList.add('hidden');
    render(r.analysis, r.suggested_questions || []);

    const analyzed = r.comments_analyzed || r.count;
    toast(`Analyzed ${fmt(analyzed)} comments out of ${fmt(r.count)} fetched`, 'ok');
    setTimeout(() => $('progress-wrap').classList.add('hidden'), 1200);

    // Start polling for RAG readiness if not already ready
    if (!S.rag) {
      startRagPolling();
    }

  } catch (e) {
    clearInterval(tick);
    toast(e.message, 'err');
    $('loading').classList.add('hidden');
    $('welcome').classList.remove('hidden');
    $('progress-wrap').classList.add('hidden');
  } finally {
    btn.disabled = false; btn.textContent = 'Analyze';
  }
}

// ─── RAG Status Polling ─────────────────────────────────────────
function startRagPolling() {
  // Show building banner
  $('rag-building-banner')?.classList.remove('hidden');
  $('chat-input').placeholder = 'Search index is building...';
  $('chat-input').disabled = true;

  if (ragPollTimer) clearInterval(ragPollTimer);
  ragPollTimer = setInterval(async () => {
    try {
      const r = await api(`/api/rag/status/${S.sid}`);
      if (r.ready) {
        clearInterval(ragPollTimer);
        ragPollTimer = null;
        S.rag = true;
        $('rag-building-banner')?.classList.add('hidden');
        $('chat-input').placeholder = 'Ask a question...';
        $('chat-input').disabled = false;
        toast('Search index ready — you can now chat!', 'ok');
      } else if (r.status === 'error') {
        clearInterval(ragPollTimer);
        ragPollTimer = null;
        $('rag-building-banner')?.classList.add('hidden');
        $('chat-input').placeholder = 'Index failed. Try re-analyzing.';
        toast('Failed to build search index: ' + (r.error || 'Unknown error'), 'err');
      }
    } catch (e) {
      // Ignore poll errors, keep trying
    }
  }, 2000);
}

// ─── Render Analysis ────────────────────────────────────────────
function render(a, questions) {
  // CRITICAL FIX: hide welcome before showing results
  $('welcome').classList.add('hidden');
  $('loading').classList.add('hidden');
  $('results').classList.remove('hidden');

  // Render metadata header
  renderMetaHeader();

  const c = $('analysis-content');
  c.innerHTML = '';
  if (!a) return;

  // Summary
  if (a.summary) {
    const html = typeof marked !== 'undefined' ? marked.parse(a.summary) : '<p>' + esc(a.summary) + '</p>';
    let s = `<div class="section"><h3>📋 Executive Summary</h3><div class="card">${html}</div>`;
    if (a.key_themes?.length)
      s += `<h4>Key Themes</h4><div class="tags">${a.key_themes.map(t => `<span class="tag">${esc(t)}</span>`).join('')}</div>`;
    if (a.notable_opinions?.length)
      s += `<h4>Notable Opinions</h4><ul>${a.notable_opinions.map(o => `<li>${esc(o)}</li>`).join('')}</ul>`;
    if (a.controversies?.length)
      s += `<h4>Controversies</h4><ul class="controversies-list">${a.controversies.map(x => `<li>${esc(x)}</li>`).join('')}</ul>`;
    s += '</div>';
    c.innerHTML += s;
  }

  // Sentiment
  if (a.sentiment) {
    const { positive: p, negative: n, neutral: u } = a.sentiment;
    const t = (p||0) + (n||0) + (u||0);
    const pf = v => t > 0 ? ((v/t)*100).toFixed(1) : '0';
    const pid = uid('pie');

    c.innerHTML += `<div class="section"><h3>📊 Sentiment Analysis</h3>
      <div class="stats-row">
        <div class="stat-card">
          <span class="stat-value">${fmt(t)}</span>
          <span class="stat-label">Analyzed</span>
        </div>
        <div class="stat-card stat-pos">
          <span class="stat-value">${pf(p)}%</span>
          <span class="stat-label">Positive</span>
        </div>
        <div class="stat-card stat-neg">
          <span class="stat-value">${pf(n)}%</span>
          <span class="stat-label">Negative</span>
        </div>
        <div class="stat-card stat-neu">
          <span class="stat-value">${pf(u)}%</span>
          <span class="stat-label">Neutral</span>
        </div>
      </div>
      <div id="${pid}" class="chart-box"></div>
    </div>`;

    requestAnimationFrame(() => {
      const el = document.getElementById(pid);
      if (!el) return;
      Plotly.newPlot(pid, [{
        type:'pie', values:[p,n,u], labels:['Positive','Negative','Neutral'],
        marker:{colors:['#22c55e','#ef4444','#6b7280']},
        textinfo:'label+percent', textfont:{color:'#fff',size:13}, hole:0.45
      }], {
        paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)',
        font:{color:'#a1a1aa',family:'Inter'}, margin:{l:20,r:20,t:10,b:10},
        showlegend:false, height:280
      }, {responsive:true, displayModeBar:false});
    });
  }

  // Topics
  if (a.topics?.length) {
    const bid = uid('bar');
    // Sort topics descending by count and reverse for Plotly horizontal bar (bottom-to-top)
    const sorted = [...a.topics].sort((a,b) => a.count - b.count);
    const names = sorted.map(t => t.name);
    const counts = sorted.map(t => t.count);
    const descs = sorted.map(t => t.description || '');

    c.innerHTML += `<div class="section"><h3>🏷️ Topics Discovered</h3>
      <div class="topics-list">${a.topics.sort((a,b) => b.count - a.count).map(t =>
        `<div class="topic-item"><span class="topic-name">${esc(t.name)}</span><span class="topic-count">${t.count}</span><span class="topic-desc">${esc(t.description)}</span></div>`
      ).join('')}</div>
      <div id="${bid}" class="chart-box" style="margin-top:12px;"></div>
    </div>`;

    requestAnimationFrame(() => {
      const el = document.getElementById(bid);
      if (!el) return;
      Plotly.newPlot(bid, [{
        type:'bar', x:counts, y:names, orientation:'h',
        marker:{color:'#3b82f6', line:{color:'#2563eb',width:1}},
        text:counts.map(String), textposition:'outside', textfont:{color:'#a1a1aa',size:12},
        hovertext:descs, hoverinfo:'text'
      }], {
        paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)',
        font:{color:'#a1a1aa',family:'Inter'},
        xaxis:{showgrid:false,showticklabels:false,zeroline:false},
        yaxis:{automargin:true, tickfont:{size:12}},
        margin:{l:10,r:50,t:10,b:10},
        height:Math.max(200, names.length*44), bargap:0.25
      }, {responsive:true, displayModeBar:false});
    });
  }

  // Chat chips
  const qs = questions.length ? questions : ["What are the main complaints?", "What do people like?", "What's controversial?", "Summarize the top opinions"];
  const chips = $('chips');
  chips.innerHTML = '';
  qs.forEach(q => {
    const d = document.createElement('div');
    d.className = 'chip';
    d.textContent = q;
    d.onclick = () => ask(q);
    chips.appendChild(d);
  });
}

function renderMetaHeader() {
  const mh = $('meta-header');
  if (!S.meta || !mh) { if(mh) mh.innerHTML = ''; return; }

  const m = S.meta;
  const platform = (S.plat||'').charAt(0).toUpperCase() + (S.plat||'').slice(1);
  let html = `<div class="meta-card">`;
  html += `<div class="meta-platform"><span class="badge badge-${S.plat === 'youtube' ? 'red' : 'orange'}">${esc(platform)}</span></div>`;

  if (m.title) html += `<h2 class="meta-title">${esc(m.title)}</h2>`;

  html += `<div class="meta-stats">`;
  if (m.channel) html += `<span class="meta-stat">📺 ${esc(m.channel)}</span>`;
  if (m.subreddit) html += `<span class="meta-stat">🗂️ r/${esc(m.subreddit)}</span>`;
  if (m.author) html += `<span class="meta-stat">👤 ${esc(m.author)}</span>`;
  if (m.view_count) html += `<span class="meta-stat">👁️ ${fmt(m.view_count)} views</span>`;
  if (m.like_count) html += `<span class="meta-stat">👍 ${fmt(m.like_count)} likes</span>`;
  if (m.score) html += `<span class="meta-stat">⬆️ ${fmt(m.score)} score</span>`;
  if (m.comment_count || m.num_comments) html += `<span class="meta-stat">💬 ${fmt(m.comment_count || m.num_comments)} comments</span>`;
  html += `</div>`;

  html += `<div class="meta-analyzed">✅ ${fmt(S.count)} comments fetched and analyzed</div>`;
  html += `</div>`;
  mh.innerHTML = html;
}

// ─── Chat ───────────────────────────────────────────────────────
function send() {
  const inp = $('chat-input');
  const q = inp.value.trim();
  if (!q) return;
  inp.value = '';
  ask(q);
}

async function ask(q) {
  if (!S.rag) return toast('Search index is still building. Please wait...', 'warn');
  $('chat-welcome').style.display = 'none';
  addMsg('user', q);
  const tid = addTyping();

  try {
    const r = await api('/api/rag/ask', { method: 'POST', body: { question: q, session_id: S.sid } });
    $(tid)?.remove();
    addMsg('ai', r.answer, r.sources);
  } catch (e) {
    $(tid)?.remove();
    if (e.message.includes('still building')) {
      addMsg('ai', '⏳ The search index is still being built. Please wait a moment and try again.');
    } else {
      addMsg('ai', '❌ Error: ' + e.message);
    }
  }
}

function addMsg(role, text, sources) {
  const box = $('chat-messages');
  const d = document.createElement('div');
  d.className = role === 'user' ? 'msg msg-user' : 'msg msg-ai';

  let head = '';
  if (role === 'ai') head = `<div class="msg-head"><div class="avatar avatar-ai">AI</div><span class="muted" style="font-size:11px">Assistant</span></div>`;

  let body = '';
  if (role === 'ai' && typeof marked !== 'undefined') body = marked.parse(text);
  else body = esc(text).replace(/\n/g, '<br>');

  let srcHtml = '';
  if (sources?.length) {
    const srcId = uid('src');
    srcHtml = `<div class="sources">
      <div class="sources-toggle" onclick="document.getElementById('${srcId}').classList.toggle('open')">▸ Sources (${sources.length})</div>
      <div id="${srcId}" class="sources-list">${sources.map((s,i) =>
        `<div class="source-item">
          <div class="source-meta">[${i+1}] ${esc(s.author || 'Unknown')}${s.likes ? ' · 👍 '+s.likes : ''} · Relevance: ${(s.relevance_score||0).toFixed(2)}</div>
          <div class="source-text">${esc((s.text||'').slice(0,300))}</div>
        </div>`
      ).join('')}</div></div>`;
  }

  d.innerHTML = `${head}<div class="msg-body">${body}${srcHtml}</div>`;
  box.appendChild(d);
  box.scrollTop = box.scrollHeight;
}

function addTyping() {
  const box = $('chat-messages');
  const id = uid('typing');
  const d = document.createElement('div');
  d.id = id; d.className = 'msg msg-ai';
  d.innerHTML = `<div class="msg-head"><div class="avatar avatar-ai">AI</div></div><div class="msg-body"><div class="typing"><span></span><span></span><span></span></div></div>`;
  box.appendChild(d);
  box.scrollTop = box.scrollHeight;
  return id;
}

// ─── Clear Chat ─────────────────────────────────────────────────
async function clearChat() {
  try {
    await api('/api/rag/clear', { method: 'POST', body: { session_id: S.sid } });
    const box = $('chat-messages');
    box.innerHTML = '';
    // Re-show welcome
    const welcome = document.createElement('div');
    welcome.id = 'chat-welcome';
    welcome.className = 'chat-welcome';
    welcome.innerHTML = `<p class="muted">Ask anything about the comments</p><div id="chips" class="chips"></div>`;
    box.appendChild(welcome);
    // Re-render chips
    const qs = ["What are the main complaints?", "What do people like?", "What's controversial?", "Summarize the top opinions"];
    const chips = $('chips');
    qs.forEach(q => {
      const d = document.createElement('div');
      d.className = 'chip';
      d.textContent = q;
      d.onclick = () => ask(q);
      chips.appendChild(d);
    });
    toast('Chat history cleared', 'ok');
  } catch (e) {
    toast('Failed to clear chat: ' + e.message, 'err');
  }
}

// ─── Export CSV ─────────────────────────────────────────────────
function exportCSV() {
  window.open(`/api/export/csv?session_id=${S.sid}`, '_blank');
}
