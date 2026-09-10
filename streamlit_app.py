import streamlit as st

# ── Page config ─────────────────────────────────────────────

st.set_page_config(
    page_title="GBG RAG Bot",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --bg-primary: #0f0f13;
    --bg-secondary: #1a1a24;
    --bg-card: #1e1e2e;
    --bg-glass: rgba(30, 30, 46, 0.7);
    --accent-primary: #7c6aef;
    --accent-secondary: #a78bfa;
    --accent-glow: rgba(124, 106, 239, 0.15);
    --text-primary: #e4e4ef;
    --text-secondary: #9090a8;
    --text-muted: #5e5e76;
    --border: rgba(124, 106, 239, 0.15);
    --success: #34d399;
    --warning: #fbbf24;
    --error: #f87171;
    --info: #60a5fa;
    --font-arabic: 'Cairo', 'Segoe UI', Tahoma, sans-serif;
    --font-latin: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

html, body, [class*="css"] {
    font-family: var(--font-latin);
}

.stApp {
    background: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

/* Sidebar styling - strictly English and LTR */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #13131d 0%, #1a1a28 100%) !important;
    border-right: 1px solid var(--border) !important;
    direction: ltr !important;
    text-align: left !important;
}

section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--text-primary) !important;
    direction: ltr !important;
    text-align: left !important;
}

/* Chat container & messages */
.stChatMessage {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 1.1rem !important;
    margin-bottom: 0.85rem !important;
    backdrop-filter: blur(10px) !important;
}

/* RTL styling for chat message contents (Arabic queries and answers) */
div[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] {
    direction: rtl;
    text-align: right;
    unicode-bidi: plaintext;
    font-family: var(--font-arabic) !important;
    font-size: 0.96rem;
    line-height: 1.85;
}

div[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] p {
    direction: rtl;
    text-align: right;
    unicode-bidi: plaintext;
    margin-bottom: 0.5rem;
}

div[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] ul,
div[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] ol {
    direction: rtl;
    text-align: right;
    padding-right: 1.5rem;
    padding-left: 0;
    margin: 0.5rem 0;
}

div[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] li {
    direction: rtl;
    text-align: right;
    unicode-bidi: plaintext;
    margin-bottom: 0.3rem;
}

div[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] h1,
div[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] h2,
div[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] h3,
div[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] h4 {
    direction: rtl;
    text-align: right;
    font-family: var(--font-arabic) !important;
    font-weight: 700;
    color: var(--accent-secondary);
}

/* Code & technical terms: isolate in LTR to prevent punctuation inversion */
div[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] code,
code {
    direction: ltr !important;
    unicode-bidi: isolate !important;
    display: inline-block !important;
    font-family: 'Fira Code', 'Consolas', monospace !important;
    font-size: 0.84em !important;
    background: rgba(124, 106, 239, 0.15) !important;
    color: var(--accent-secondary) !important;
    padding: 0.12rem 0.4rem !important;
    border-radius: 4px !important;
    border: 1px solid rgba(124, 106, 239, 0.25) !important;
}

pre, pre code {
    direction: ltr !important;
    text-align: left !important;
    unicode-bidi: isolate !important;
    display: block !important;
    background: #13131d !important;
    border-radius: 8px !important;
    padding: 0.8rem !important;
}

/* Chat Input: RTL text alignment with Arabic font */
div[data-testid="stChatInput"] {
    border-color: var(--border) !important;
}

div[data-testid="stChatInput"] textarea {
    direction: rtl !important;
    text-align: right !important;
    unicode-bidi: plaintext !important;
    font-family: var(--font-arabic) !important;
    font-size: 0.95rem !important;
    color: var(--text-primary) !important;
}

div[data-testid="stChatInput"] textarea::placeholder {
    direction: rtl !important;
    text-align: right !important;
    color: var(--text-muted) !important;
    font-family: var(--font-arabic) !important;
}

/* Expander styling - strictly LTR headers for English concepts */
div[data-testid="stExpander"] {
    direction: ltr !important;
    text-align: left !important;
}

.streamlit-expanderHeader {
    background: var(--bg-secondary) !important;
    color: var(--text-primary) !important;
    border-radius: 8px !important;
    border: 1px solid var(--border) !important;
    direction: ltr !important;
    text-align: left !important;
    font-family: var(--font-latin) !important;
}

.streamlit-expanderContent {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border) !important;
    border-top: none !important;
    border-radius: 0 0 8px 8px !important;
}

/* Metric cards - strictly LTR for English labels & numbers */
.metric-card {
    background: linear-gradient(135deg, var(--bg-card) 0%, rgba(30,30,46,0.5) 100%);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
    direction: ltr;
    backdrop-filter: blur(12px);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px var(--accent-glow);
}

.metric-label {
    font-size: 0.72rem;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.25rem;
    font-family: var(--font-latin);
    direction: ltr;
}

.metric-value {
    font-size: 1.35rem;
    font-weight: 700;
    color: var(--accent-secondary);
    line-height: 1.2;
    font-family: var(--font-latin);
    direction: ltr;
}

.metric-value.success { color: var(--success); }
.metric-value.warning { color: var(--warning); }
.metric-value.info    { color: var(--info); }
.metric-value.error   { color: var(--error); }

/* Route badge & technique pill */
.route-badge {
    display: inline-block;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    background: linear-gradient(135deg, var(--accent-primary), #9b6aef);
    color: #fff;
    box-shadow: 0 2px 12px var(--accent-glow);
    font-family: var(--font-latin);
    direction: ltr;
}

.technique-pill {
    display: inline-block;
    padding: 0.25rem 0.7rem;
    border-radius: 16px;
    font-size: 0.75rem;
    font-weight: 500;
    background: rgba(96, 165, 250, 0.12);
    color: var(--info);
    border: 1px solid rgba(96, 165, 250, 0.2);
    margin: 0.15rem 0.2rem;
    font-family: var(--font-latin);
    direction: ltr;
}

/* Chunk card - Arabic body with proper RTL + bidi, English metadata */
.chunk-card {
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem;
    margin: 0.5rem 0;
    font-size: 0.88rem;
    line-height: 1.85;
    direction: rtl;
    text-align: right;
    unicode-bidi: plaintext;
    font-family: var(--font-arabic);
}

.chunk-meta {
    font-size: 0.72rem;
    color: var(--text-muted);
    margin-bottom: 0.5rem;
    direction: ltr !important;
    text-align: left !important;
    unicode-bidi: isolate !important;
    font-family: var(--font-latin), monospace;
}

/* Header */
.app-header {
    text-align: center;
    padding: 1.5rem 0 1rem;
    direction: ltr;
}

.app-header h1 {
    font-size: 1.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, var(--accent-secondary), #c084fc, var(--info));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.25rem;
    font-family: var(--font-latin);
}

.app-header p {
    color: var(--text-muted);
    font-size: 0.85rem;
    font-family: var(--font-latin);
}

/* Divider */
.section-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border), transparent);
    margin: 1rem 0;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--text-muted); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-secondary); }

/* Buttons */
.stButton button {
    background: linear-gradient(135deg, var(--accent-primary), #9b6aef) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-family: var(--font-latin) !important;
    transition: all 0.2s ease !important;
}

.stButton button:hover {
    box-shadow: 0 4px 20px var(--accent-glow) !important;
    transform: translateY(-1px) !important;
}

/* Hide default Streamlit elements */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Helper functions ────────────────────────────────────────

def _estimate_cost(
    input_tokens: int,
    output_tokens: int,
) -> float:
    """
    Rough cost estimate based on Groq pricing.
    """
    input_rate = 0.0005 / 1000
    output_rate = 0.0015 / 1000
    return (
        input_tokens * input_rate
        + output_tokens * output_rate
    )


def _render_result_panels(result: dict):
    """Render the metrics, routing info, and log panels
    for an assistant message."""

    st.markdown(
        '<div class="section-divider"></div>',
        unsafe_allow_html=True,
    )

    # ── Metric cards row ────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">⏱️ Total Time</div>
            <div class="metric-value">{result['execution_time_s']}s</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        cost = _estimate_cost(
            result["estimated_input_tokens"],
            result["estimated_output_tokens"],
        )
        color_class = "success" if cost < 0.01 else "warning"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">💰 Est. Cost</div>
            <div class="metric-value {color_class}">${cost:.4f}</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">📥 Input Tokens</div>
            <div class="metric-value info">~{result['estimated_input_tokens']}</div>
        </div>""", unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">📤 Output Tokens</div>
            <div class="metric-value info">~{result['estimated_output_tokens']}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")

    # ── Router decision ─────────────────────────────────
    r1, r2 = st.columns([1, 2])

    with r1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🧭 Router Decision</div>
            <div style="margin-top: 0.4rem;">
                <span class="route-badge">{result['route']}</span>
            </div>
        </div>""", unsafe_allow_html=True)

    with r2:
        fallback_html = ""
        if result.get("fallback_used"):
            fallback_html = (
                '<div style="margin-top:0.2rem; font-size:0.75rem; '
                'color:#f87171;">⚠️ Fallback to basic retrieval was used</div>'
            )

        st.markdown(f"""
        <div class="metric-card" style="text-align: left;">
            <div class="metric-label">🔧 Technique Executed</div>
            <div style="margin-top: 0.3rem;">
                <span class="technique-pill">{result['technique']}</span>
            </div>
            <div style="margin-top: 0.4rem; font-size: 0.78rem; color: #9090a8; direction: auto; unicode-bidi: plaintext; text-align: start;">
                <strong style="direction: ltr; display: inline-block;">Reason:</strong> <span dir="auto">{result['route_reason']}</span>
            </div>
            {fallback_html}
        </div>""", unsafe_allow_html=True)

    st.markdown("")

    # ── Timing breakdown ────────────────────────────────
    with st.expander("⏱️ Timing Breakdown"):
        tc1, tc2, tc3 = st.columns(3)
        with tc1:
            st.metric(
                "Router",
                f"{result.get('route_time_s', '?')}s",
            )
        with tc2:
            st.metric(
                "Retrieval",
                f"{result.get('retrieval_time_s', '?')}s",
            )
        with tc3:
            st.metric(
                "Generation",
                f"{result.get('generation_time_s', '?')}s",
            )

    # ── Retrieved documents ─────────────────────────────
    docs = result.get("documents", [])
    with st.expander(f"📄 Retrieved Documents ({len(docs)})"):
        if not docs:
            st.info("No documents retrieved.")
        for j, doc in enumerate(docs):
            meta = doc.get("metadata", {})
            chunk_id = meta.get("chunk_id", f"chunk_{j}")
            source = meta.get("source", "unknown")
            page = meta.get("page", "?")
            text = doc.get("document", "")

            preview = text[:500]
            if len(text) > 500:
                preview += "…"

            st.markdown(f"""
            <div class="chunk-card">
                <div class="chunk-meta">
                    <strong>{chunk_id}</strong> &nbsp;·&nbsp;
                    📁 {source} &nbsp;·&nbsp;
                    📄 Page {page}
                </div>
                {preview}
            </div>""", unsafe_allow_html=True)


# ── Sidebar ─────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🏦</div>
        <h2 style="margin: 0; font-size: 1.2rem; font-weight: 700;
                    background: linear-gradient(135deg, #a78bfa, #60a5fa);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;">
            GBG RAG Bot
        </h2>
        <p style="color: #5e5e76; font-size: 0.75rem; margin-top: 0.25rem;">
            Housing & Development Bank
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-divider"></div>',
        unsafe_allow_html=True,
    )

    st.markdown("### 📋 About")
    st.markdown("""
    <p style="font-size: 0.82rem; color: #9090a8; line-height: 1.7;">
        An advanced Arabic RAG system for querying internal bank
        documentation. Uses hybrid retrieval (Dense + BM25),
        intelligent routing, and multiple advanced techniques.
    </p>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-divider"></div>',
        unsafe_allow_html=True,
    )

    st.markdown("### ⚡ Available Techniques")
    techniques = [
        "Basic Hybrid", "Query Rewriting", "Multi-Query",
        "Decomposition", "HyDE", "Self-Query",
        "Reranking", "Compression", "CRAG",
    ]
    for t in techniques:
        st.markdown(
            f'<span class="technique-pill">{t}</span>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="section-divider"></div>',
        unsafe_allow_html=True,
    )

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.results = []
        st.rerun()


# ── Session state ───────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

if "results" not in st.session_state:
    st.session_state.results = []


# ── Header ──────────────────────────────────────────────────

st.markdown("""
<div class="app-header">
    <h1>🏦 GBG RAG Bot</h1>
    <p>Advanced Arabic RAG System · Housing & Development Bank</p>
</div>
""", unsafe_allow_html=True)


# ── Chat history ────────────────────────────────────────────

result_idx = 0

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if (
            msg["role"] == "assistant"
            and result_idx < len(st.session_state.results)
        ):
            _render_result_panels(
                st.session_state.results[result_idx]
            )
            result_idx += 1


# ── Chat input & processing ────────────────────────────────

if prompt := st.chat_input(
    "اسأل سؤالاً عن وثائق البنك..."
):
    # Display user message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    # Process with the RAG pipeline
    with st.chat_message("assistant"):
        with st.spinner("🔍 Processing your question..."):
            from advanced_rag.generator import (
                generate_advanced_answer_with_metadata,
            )
            from basic_rag.retrieval.retriever import (
                get_collection_info,
            )
            from basic_rag.indexing.embedder import (
                generate_and_index_embeddings,
            )

            # Ensure collection is populated
            info = get_collection_info()
            if info["count"] == 0:
                with st.status(
                    "📚 Indexing documents...",
                    expanded=True,
                ):
                    generate_and_index_embeddings()

            result = generate_advanced_answer_with_metadata(
                prompt
            )

        # Display answer
        st.markdown(result["answer"])

        # Store for history
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
        })
        st.session_state.results.append(result)

        # Render panels
        _render_result_panels(result)
