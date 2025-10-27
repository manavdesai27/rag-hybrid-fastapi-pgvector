"""Streamlit UI for the RAG demo with chat-style input/output.

- Sidebar controls (API URL, max tokens, show raw, clear chat)
- Chat bubbles using st.chat_message
- Single-turn backend call to /ask (UI stores history locally)
- Shows answer, citations, and a small metrics caption per assistant reply
"""
import os
import time
import json
import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="RAG Demo (SBC Docs)", page_icon="🧭", layout="wide")

# Initialize chat history
if "messages" not in st.session_state:
    # Each item: {"role": "user"|"assistant", "content": str}
    st.session_state.messages = []
if "thinking" not in st.session_state:
    st.session_state.thinking = False
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

st.title("RAG Demo — Hybrid Search + Citations (SBC Docs)")
st.caption(
    "Searches SBC docs first (keyword + semantic), then answers grounded on retrieved passages with citations."
)

with st.sidebar:
    st.subheader("Settings")
    api_url = st.text_input("API Base URL", value=API_BASE_URL, help="Backend FastAPI base URL")
    show_raw = st.checkbox("Show raw response", value=False)
    max_tokens = st.slider(
        "Max answer tokens",
        min_value=256,
        max_value=8192,
        value=256,
        step=64,
        help="Upper bound on completion length",
    )
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.session_state.thinking = False
        st.session_state.pending_prompt = None
        st.rerun()

def health_check(url: str) -> bool:
    """Return True if the backend health endpoint responds OK."""
    try:
        r = requests.get(f"{url}/health")
        return r.ok
    except Exception as e:
        # Show the exception only as info to avoid alarming UX for transient issues.
        st.info(e)
        return False


ok = health_check(api_url)
if not ok:
    st.warning(
        f"Backend health check failed at {api_url}/health. "
        "If running via Docker, ensure 'docker compose -f infra/docker-compose.yml up' is running and ingestion is done."
    )

# Render existing chat history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        # Support both legacy simple messages (content only) and structured assistant messages
        if m.get("role") == "assistant" and ("answer" in m):
            st.markdown(m.get("answer") or "_No answer returned._")
            latency_ms = m.get("latency_ms", None)
            used_cache_bool = m.get("used_cache", None)
            if latency_ms is not None and used_cache_bool is not None:
                used_cache_label = "Yes" if used_cache_bool else "No"
                st.caption(f"Latency: {latency_ms} ms • Cache: {used_cache_label} • Citations: {len(m.get('citations', []))}")
            citations_hist = m.get("citations") or []
            if citations_hist:
                with st.expander(f"Citations ({len(citations_hist)})", expanded=False):
                    for i, c in enumerate(citations_hist, start=1):
                        title = c.get("source") or "untitled"
                        url = c.get("url")
                        if url:
                            st.markdown(f"- [{i}] [{title}]({url})")
                        else:
                            st.markdown(f"- [{i}] {title}")
                        snip = c.get("snippet")
                        if snip:
                            st.markdown(f"> {snip}")
        else:
            st.markdown(m.get("content", ""))

# Chat input — disable while thinking
if st.session_state.thinking:
    st.chat_input("Ask something about SBC ACLI...", disabled=True, key="disabled_input")
else:
    prompt = st.chat_input("Ask something about SBC ACLI...", key="enabled_input")
    if prompt:
        # Append and show user message immediately
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        # Queue prompt and enter thinking state
        st.session_state.pending_prompt = prompt
        st.session_state.thinking = True
        st.rerun()

# Process pending request when thinking; keep input disabled during this phase
if st.session_state.thinking and st.session_state.pending_prompt:
    pending = st.session_state.pending_prompt

    # Assistant bubble: call backend and render response
    with st.chat_message("assistant"):
        if not ok:
            st.info("Backend is not healthy yet. Start the stack and try again.")
        else:
            with st.spinner("Thinking..."):
                t0 = time.time()
                try:
                    resp = requests.post(
                        f"{api_url}/ask",
                        json={"question": pending, "max_tokens": int(max_tokens)},
                        timeout=90,
                    )
                    dt_ms = int((time.time() - t0) * 1000.0)
                    if not resp.ok:
                        st.error(f"Request failed: {resp.status_code} {resp.text}")
                    else:
                        # Try to parse JSON; if it fails, show raw text and stop
                        try:
                            data = resp.json()
                        except Exception:
                            st.error("Response was not valid JSON.")
                            st.code(resp.text or "", language="json")
                            data = {"answer": "", "citations": [], "used_cache": False, "latency_ms": dt_ms}

                        answer = data.get("answer", "")
                        citations = data.get("citations", []) or []
                        used_cache = "Yes" if data.get("used_cache", False) else "No"
                        latency_ms = data.get("latency_ms", dt_ms)

                        # Render answer and low-visibility citations via expander
                        st.markdown(answer if answer else "_No answer returned._")
                        st.caption(f"Latency: {latency_ms} ms • Cache: {used_cache} • Citations: {len(citations)}")
                        if citations:
                            with st.expander(f"Citations ({len(citations)})", expanded=False):
                                for i, c in enumerate(citations, start=1):
                                    title = c.get("source") or "untitled"
                                    url = c.get("url")
                                    if url:
                                        st.markdown(f"- [{i}] [{title}]({url})")
                                    else:
                                        st.markdown(f"- [{i}] {title}")
                                    snippet = c.get("snippet")
                                    if snippet:
                                        st.markdown(f"> {snippet}")

                        if show_raw:
                            try:
                                st.code(json.dumps(data, indent=2), language="json")
                            except Exception:
                                st.code(resp.text or "", language="json")

                        # Persist structured message for consistent re-render with expander
                        st.session_state.messages.append({
                            "role": "assistant",
                            "answer": answer,
                            "citations": citations,
                            "latency_ms": latency_ms,
                            "used_cache": True if used_cache == "Yes" else False,
                        })
                except Exception as e:
                    st.error(f"Error calling API: {e}")

    # Clear state and rerun to re-enable input
    st.session_state.thinking = False
    st.session_state.pending_prompt = None
    st.rerun()
