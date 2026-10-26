import os
import io
import json
import base64
import hashlib
from pathlib import Path
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

# Resilient DuckDuckGo Search Import
try:
    from duckduckgo_search import DDGS
except ImportError:
    try:
        from ddgs import DDGS
    except ImportError:
        DDGS = None

# =========================================================
# CONFIGURATION & DIRECTORIES
# =========================================================

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "images"
REPORT_DIR = ROOT / "reports"
HISTORY_DIR = ROOT / "history"

for folder in (DATA_DIR, REPORT_DIR, HISTORY_DIR):
    folder.mkdir(parents=True, exist_ok=True)

load_dotenv(ROOT / ".env")

ENV_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
DEFAULT_MODEL = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3.5-super-vl:free").strip()
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

POPULAR_VISION_MODELS = [
    "google/gemini-2.5-flash",
    "google/gemini-flash-1.5",
    "google/gemini-2.0-flash-exp:free",
    "meta-llama/llama-3.2-11b-vision-instruct:free",
    "qwen/qwen-2.5-vl-72b-instruct:free",
    "nvidia/nemotron-3.5-super-vl:free",
    "openai/gpt-4o-mini",
    "anthropic/claude-3.5-sonnet",
    "Custom Model...",
]

st.set_page_config(
    page_title="UC04 | Visual Intelligence",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# SESSION STATE INITIALIZATION
# =========================================================

PAGES = [
    "Overview",
    "Image Investigation",
    "Dataset Analysis",
    "Reports Library",
    "Investigation Chat",
]

if "active_page" not in st.session_state:
    st.session_state["active_page"] = "Overview"

if "chat_histories" not in st.session_state:
    st.session_state["chat_histories"] = {}

if "current_result" not in st.session_state:
    st.session_state["current_result"] = None

# =========================================================
# DESIGN SYSTEM & CSS
# =========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --navy: #101828;
    --navy-soft: #1D2939;
    --blue: #2563EB;
    --blue-hover: #1D4ED8;
    --blue-soft: #EFF6FF;
    --ink: #111827;
    --muted: #6B7280;
    --line: #E5E7EB;
    --surface: #FFFFFF;
    --canvas: #F9FAFB;
    --green: #15803D;
}

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

.stApp {
    background: var(--canvas);
}

.block-container {
    max-width: 1400px;
    padding-top: 1.75rem;
    padding-bottom: 3rem;
}

/* Sidebar styling without breaking form control contrast */
section[data-testid="stSidebar"] {
    background-color: var(--navy);
    border-right: 1px solid #1F2A37;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label {
    color: #E5E7EB !important;
}

section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] select {
    color: #111827 !important;
}

section[data-testid="stSidebar"] hr {
    border-color: #374151;
}

.hero {
    background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 100%);
    color: white;
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 24px;
    box-shadow: 0 4px 14px rgba(15, 23, 42, 0.08);
}

.hero-kicker {
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 2px;
    color: #93C5FD;
    text-transform: uppercase;
    margin-bottom: 8px;
}

.hero h1 {
    color: #FFFFFF !important;
    font-size: clamp(24px, 2.5vw, 34px);
    font-weight: 800;
    line-height: 1.25;
    margin: 0 0 10px 0;
}

.hero p {
    color: #DBEAFE;
    font-size: 14px;
    line-height: 1.6;
    max-width: 780px;
    margin-bottom: 0;
}

.page-heading {
    color: var(--ink);
    font-size: 24px;
    font-weight: 800;
    margin: 0 0 4px 0;
}

.page-description {
    color: var(--muted);
    font-size: 13px;
    line-height: 1.6;
    margin-bottom: 20px;
}

.section-heading {
    font-size: 16px;
    font-weight: 750;
    color: var(--ink);
    margin: 16px 0 10px 0;
}

.panel {
    background: white;
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 18px 20px;
    margin-bottom: 14px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.02);
}

.panel-title {
    color: var(--ink);
    font-size: 15px;
    font-weight: 700;
    margin-bottom: 6px;
}

.panel-description {
    color: var(--muted);
    font-size: 13px;
    line-height: 1.6;
}

.eyebrow {
    color: var(--muted);
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 6px;
}

[data-testid="stMetric"] {
    background: white;
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 14px 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.02);
}

.stButton > button {
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.15s ease-in-out;
}

.footer {
    color: var(--muted);
    font-size: 12px;
    border-top: 1px solid var(--line);
    padding-top: 18px;
    margin-top: 36px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# BACKEND UTILITIES & API CLIENT
# =========================================================

def get_openrouter_client(api_key: str):
    """Create an OpenAI SDK client configured for OpenRouter."""
    if not api_key:
        raise ValueError("OpenRouter API key is required. Enter it in the sidebar.")

    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        default_headers={
            "HTTP-Referer": "http://localhost:8501",
            "X-Title": "UC04 Visual Investigation AI",
        },
        timeout=120,
        max_retries=2,
    )


def prepare_image(source):
    """Resize image and encode to base64 JPEG data URL."""
    with Image.open(source) as original:
        image = original.convert("RGB")
        image.thumbnail((1024, 1024))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85)

    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def safe_name(filename: str) -> str:
    """Generate safe filename slug."""
    stem = Path(filename).stem
    cleaned = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)
    return cleaned[:60] or "case"


def ask_model(messages, model: str, api_key: str, max_tokens=1800) -> str:
    """Execute completion request with clear error messages."""
    client = get_openrouter_client(api_key)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.2,
        )
        answer = response.choices[0].message.content
        if not answer or not answer.strip():
            raise ValueError("The model returned an empty response.")
        return answer.strip()
    except Exception as e:
        msg = str(e)
        if "401" in msg or "Unauthorized" in msg:
            raise RuntimeError("Authentication failed. Check your OpenRouter API key.")
        elif "402" in msg:
            raise RuntimeError("Insufficient OpenRouter account credits.")
        elif "does not support image" in msg or "Invalid parameter: messages" in msg:
            raise RuntimeError(f"The model '{model}' does not accept image inputs. Select a vision model like Gemini or Qwen VL.")
        raise RuntimeError(f"AI Service Error: {msg}")


def research_web(query: str, limit=5):
    """Search DuckDuckGo for verification clues."""
    if not DDGS:
        return [{
            "title": "Search Unavailable",
            "url": "",
            "snippet": "duckduckgo-search package is not installed. Install via: pip install duckduckgo-search",
        }]
    try:
        with DDGS() as search:
            results = list(search.text(query, max_results=limit))
        return [
            {
                "title": item.get("title", "Untitled Source"),
                "url": item.get("href", ""),
                "snippet": item.get("body", ""),
            }
            for item in (results or [])
        ]
    except Exception as error:
        return [{
            "title": "External Search Error",
            "url": "",
            "snippet": f"Could not complete web search: {error}",
        }]


def investigate(image_url: str, filename: str, model: str, api_key: str, enable_research: bool):
    """Perform evidence-first visual investigation."""
    visual_prompt = f"""
You are an expert forensic visual investigator specializing in architecture, cultural heritage,
archaeology, and historical monuments.

Image filename: {filename}

Analyze this image rigorously using this order:
1. Describe what is physically visible.
2. Catalog architectural style, construction materials, inscriptions, carving, landscape, and environment.
3. Develop plausible landmark or regional hypotheses.
4. Estimate location only as specifically as visual evidence warrants.
5. Provide counter-hypotheses and alternative candidates.
6. Note any image quality or photographic limitations.

Return your findings in structured Markdown using these exact headings:

### Executive Summary
### Image Summary
### Primary Subject
### Directly Observed Clues
### Inferred Findings
### Probable Location
### Landmark Identification
### Location Confidence
### Authenticity Assessment
### Historical and Cultural Context
### Reasoning and Alternative Hypotheses
### Additional Observations
### Verification Status
### Uncertainties

Evidence Rules:
- Distinguish direct observation from inference.
- Do not claim a fact has been externally verified without explicit citation.
- If uncertain, clearly articulate what remains unknown.
"""

    initial_report = ask_model(
        [{
            "role": "user",
            "content": [
                {"type": "text", "text": visual_prompt},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }],
        model=model,
        api_key=api_key,
        max_tokens=2000,
    )

    sources = []

    # If research is disabled, we already have a complete report
    if not enable_research:
        return {
            "filename": filename,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "model": model,
            "report": initial_report,
            "sources": [],
        }

    # Step 2: Extract focused search query
    try:
        search_prompt = f"""
Based on this preliminary monument investigation, generate a single, highly specific web search query
to confirm the landmark name or distinctive architectural features. Return ONLY the search query.

Findings excerpt:
{initial_report[:2500]}
"""
        query = ask_model(
            [{"role": "user", "content": search_prompt}],
            model=model,
            api_key=api_key,
            max_tokens=80,
        ).strip().strip('"')

        if query:
            sources = research_web(query, limit=5)
    except Exception:
        sources = []

    # Step 3: Synthesis with retrieved sources
    source_text = "\n\n".join(
        f"Source: {s['title']}\nURL: {s['url']}\nSnippet: {s['snippet']}"
        for s in sources if s.get("snippet")
    ) or "No external verification sources retrieved."

    final_prompt = f"""
Refine and finalize the visual investigation report using external search findings.

PRELIMINARY INVESTIGATION:
{initial_report}

EXTERNAL WEB FINDINGS:
{source_text}

Update the report in Markdown using the same standard headings:
### Executive Summary
### Image Summary
### Primary Subject
### Directly Observed Clues
### Inferred Findings
### Probable Location
### Landmark Identification
### Location Confidence
### Authenticity Assessment
### Historical and Cultural Context
### Reasoning and Alternative Hypotheses
### Additional Observations
### Verification Status
### Uncertainties

Rules:
- Under 'Verification Status', explicitly note which claims were supported by the external sources.
- List relevant source URLs.
- If findings remain unverified or disputed, state that transparently.
"""

    final_report = ask_model(
        [{"role": "user", "content": final_prompt}],
        model=model,
        api_key=api_key,
        max_tokens=2400,
    )

    return {
        "filename": filename,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": model,
        "report": final_report,
        "sources": sources,
    }


def save_report(result, image_bytes=None):
    """Save report JSON and persist image to disk."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    digest = hashlib.sha1((result["filename"] + result["created_at"]).encode()).hexdigest()[:6]
    base_stem = f"{safe_name(result['filename'])}_{timestamp}_{digest}"

    # If image bytes provided, persist to DATA_DIR so other tabs can load it
    local_image_name = f"{base_stem}.jpg"
    target_img_path = DATA_DIR / local_image_name
    if image_bytes and not target_img_path.exists():
        try:
            with Image.open(io.BytesIO(image_bytes)) as img:
                img.convert("RGB").save(target_img_path, "JPEG", quality=90)
            result["saved_image"] = local_image_name
        except Exception:
            pass

    path = REPORT_DIR / f"{base_stem}.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    with (HISTORY_DIR / "investigations.jsonl").open("a", encoding="utf-8") as file:
        file.write(json.dumps(result, ensure_ascii=False) + "\n")

    return path


def load_reports():
    """Load existing reports ordered by newest first."""
    reports = []
    for path in sorted(REPORT_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            reports.append((path, json.loads(path.read_text(encoding="utf-8"))))
        except (ValueError, OSError):
            continue
    return reports


def get_image_for_report(report_data):
    """Retrieve saved image path or fallback from dataset images."""
    if "saved_image" in report_data and (DATA_DIR / report_data["saved_image"]).exists():
        return DATA_DIR / report_data["saved_image"]
    original_fn = report_data.get("filename", "")
    if (DATA_DIR / original_fn).exists():
        return DATA_DIR / original_fn
    return None


# =========================================================
# SIDEBAR
# =========================================================

dataset_images = sorted(
    p for p in DATA_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in EXTENSIONS and not p.stem.endswith("_thumb")
)
all_saved_reports = load_reports()

with st.sidebar:
    st.markdown(
        "<div style='font-size:11px;font-weight:800;letter-spacing:1.5px;color:#93C5FD'>"
        "FORENSIC INTELLIGENCE</div>"
        "<div style='font-size:20px;font-weight:800;color:white;margin:4px 0 14px'>"
        "Visual Investigation</div>",
        unsafe_allow_html=True,
    )

    # Clean navigation synced with state
    def on_nav_change():
        st.session_state["active_page"] = st.session_state["nav_selection"]

    current_idx = PAGES.index(st.session_state["active_page"]) if st.session_state["active_page"] in PAGES else 0
    st.radio(
        "WORKSPACE NAVIGATION",
        PAGES,
        index=current_idx,
        key="nav_selection",
        on_change=on_nav_change,
    )

    st.divider()

    st.markdown("<div class='eyebrow' style='color:#9CA3AF'>AI Configuration</div>", unsafe_allow_html=True)
    
    # API Key Input with .env fallback
    api_key_input = st.text_input(
        "OpenRouter API Key",
        value=ENV_API_KEY,
        type="password",
        placeholder="sk-or-v1-...",
        help="Reads from .env by default. Enter or override key here.",
    ).strip()

    # Model Selection
    model_choice = st.selectbox(
        "Vision Model",
        POPULAR_VISION_MODELS,
        index=0 if DEFAULT_MODEL not in POPULAR_VISION_MODELS else POPULAR_VISION_MODELS.index(DEFAULT_MODEL),
    )

    if model_choice == "Custom Model...":
        active_model = st.text_input("Enter model ID", value=DEFAULT_MODEL).strip()
    else:
        active_model = model_choice

    enable_research = st.toggle("Enable External Web Verification", value=True)

    st.divider()

    # Status badges
    if api_key_input:
        st.success("API key connected", icon="✅")
    else:
        st.warning("API key missing. Enter key above.", icon="⚠️")

    st.caption(f"📁 {len(dataset_images)} dataset images stored")
    st.caption(f"📑 {len(all_saved_reports)} investigation reports archived")


# =========================================================
# PAGE HEADER HERO
# =========================================================

st.markdown("""
<div class="hero">
  <div class="hero-kicker">UC04 · EVIDENCE-FIRST VISUAL INTELLIGENCE</div>
  <h1>Forensic Landmark & Architectural Investigation</h1>
  <p>Analyze cultural sites and monuments using visual evidence, architectural features,
  and cross-referenced external search verification.</p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# ROUTE 1: OVERVIEW
# =========================================================

if st.session_state["active_page"] == "Overview":
    st.markdown('<div class="page-heading">Workspace Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-description">Quick access to tools, statistics, and recent case files.</div>', unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Local Dataset Images", len(dataset_images))
    m2.metric("Saved Reports", len(all_saved_reports))
    m3.metric(
        "Generated Today",
        sum(1 for _, r in all_saved_reports if r.get("created_at", "")[:10] == datetime.now().date().isoformat()),
    )
    m4.metric("Active Model", active_model.split("/")[-1])

    st.markdown('<div class="section-heading">Quick Actions</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("""
        <div class="panel">
          <div class="eyebrow">01 · SINGLE INVESTIGATION</div>
          <div class="panel-title">Analyze a Monument</div>
          <div class="panel-description">
            Upload an image or select a dataset photo to run visual analysis and external research.
          </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Start Image Investigation", type="primary", use_container_width=True):
            st.session_state["active_page"] = "Image Investigation"
            st.rerun()

    with c2:
        st.markdown("""
        <div class="panel">
          <div class="eyebrow">02 · BATCH PROCESSING</div>
          <div class="panel-title">Investigate Image Dataset</div>
          <div class="panel-description">
            Batch analyze the image directory to generate case reports automatically.
          </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open Dataset Analysis", use_container_width=True):
            st.session_state["active_page"] = "Dataset Analysis"
            st.rerun()

    st.markdown('<div class="section-heading">Recent Investigations</div>', unsafe_allow_html=True)
    if not all_saved_reports:
        st.info("No investigations recorded yet. Click 'Start Image Investigation' to analyze your first image.")
    else:
        for path, item in all_saved_reports[:4]:
            with st.expander(f"📁 {item.get('filename', path.stem)} — {item.get('created_at', '')}"):
                img_path = get_image_for_report(item)
                if img_path:
                    st.image(str(img_path), width=320)
                st.markdown(item.get("report", ""))
                st.download_button(
                    "Download JSON Report",
                    data=json.dumps(item, indent=2, ensure_ascii=False),
                    file_name=path.name,
                    mime="application/json",
                    key=f"overview_{path.name}",
                )


# =========================================================
# ROUTE 2: IMAGE INVESTIGATION
# =========================================================

elif st.session_state["active_page"] == "Image Investigation":
    st.markdown('<div class="page-heading">Image Investigation</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-description">Inspect visual clues, test location hypotheses, and produce verified reports.</div>', unsafe_allow_html=True)

    left, right = st.columns([1.1, 0.9], gap="large")

    with left:
        source_mode = st.radio("Image Source", ["Upload File", "Select from Dataset"], horizontal=True)

        source = None
        filename = None
        raw_bytes = None

        if source_mode == "Upload File":
            uploaded = st.file_uploader(
                "Upload a monument or architectural image",
                type=["jpg", "jpeg", "png", "webp", "bmp"],
            )
            if uploaded is not None:
                source = uploaded
                filename = uploaded.name
                raw_bytes = uploaded.getvalue()
        else:
            if not dataset_images:
                st.info(f"No images found in {DATA_DIR}. Upload an image above to populate.")
            else:
                chosen_img = st.selectbox("Choose an image from data/images", [p.name for p in dataset_images])
                source = DATA_DIR / chosen_img
                filename = chosen_img
                raw_bytes = source.read_bytes()

        if source is not None:
            st.image(source, caption=f"Target: {filename}", use_container_width=True)

        can_run = source is not None and bool(api_key_input) and bool(active_model)
        run_analysis = st.button(
            "Execute Investigation",
            type="primary",
            use_container_width=True,
            disabled=not can_run,
        )

        if not api_key_input:
            st.caption("⚠️ Please supply an OpenRouter API key in the sidebar to enable analysis.")

    with right:
        st.markdown("""
        <div class="panel">
          <div class="panel-title">Investigation Methodology</div>
          <div class="panel-description">
            <b>1. Visual Feature Cataloging</b><br>
            Materials, masonry style, dome/spire architecture, weathering, and environment.<br><br>
            <b>2. Hypotheses Formation</b><br>
            Regional architectural comparisons and candidate sites.<br><br>
            <b>3. External Verification</b><br>
            Targeted search queries to validate specific landmarks without confirmation bias.<br><br>
            <b>4. Evidence Segregation</b><br>
            Strict separation between direct visual observations and inferred claims.
          </div>
        </div>
        """, unsafe_allow_html=True)

    if run_analysis and source is not None:
        try:
            with st.spinner("Analyzing image features, extracting search queries, and synthesizing report..."):
                img_data_url = prepare_image(source)
                result = investigate(
                    image_url=img_data_url,
                    filename=filename,
                    model=active_model,
                    api_key=api_key_input,
                    enable_research=enable_research,
                )
                save_report(result, image_bytes=raw_bytes)

            st.session_state["current_result"] = result
            st.success("Investigation report successfully completed!")
        except Exception as err:
            st.error(f"Investigation Error: {err}")

    # Display Active Report
    res = st.session_state.get("current_result")
    if res:
        st.divider()
        st.markdown('<div class="page-heading">Investigation Findings</div>', unsafe_allow_html=True)
        
        info_c1, info_c2, info_c3 = st.columns(3)
        info_c1.metric("Case File", res.get("filename", "N/A"))
        info_c2.metric("Timestamp", res.get("created_at", "N/A"))
        info_c3.metric("Investigated With", res.get("model", "N/A").split("/")[-1])

        tab_rep, tab_sources, tab_raw, tab_export = st.tabs([
            "📋 Formatted Report",
            "🌐 Web References",
            "📝 Raw Markdown",
            "💾 Export",
        ])

        with tab_rep:
            st.markdown(res.get("report", ""))

        with tab_sources:
            sources = res.get("sources", [])
            if not sources:
                st.info("No external search sources were collected for this case.")
            else:
                for idx, src in enumerate(sources, start=1):
                    with st.container(border=True):
                        st.markdown(f"**{idx}. {src.get('title', 'Source')}**")
                        st.caption(src.get("snippet", "No preview snippet available."))
                        if src.get("url"):
                            st.markdown(f"[🔗 Open Source Link]({src['url']})")

        with tab_raw:
            st.text_area("Markdown Content", value=res.get("report", ""), height=350)

        with tab_export:
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.download_button(
                    "Download Report as JSON",
                    data=json.dumps(res, indent=2, ensure_ascii=False),
                    file_name=f"{safe_name(res['filename'])}_report.json",
                    mime="application/json",
                    use_container_width=True,
                )
            with col_d2:
                st.download_button(
                    "Download Report as Markdown",
                    data=res.get("report", ""),
                    file_name=f"{safe_name(res['filename'])}_report.md",
                    mime="text/markdown",
                    use_container_width=True,
                )


# =========================================================
# ROUTE 3: DATASET BATCH ANALYSIS
# =========================================================

elif st.session_state["active_page"] == "Dataset Analysis":
    st.markdown('<div class="page-heading">Dataset Batch Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-description">Process multiple dataset images sequentially with progress tracking.</div>', unsafe_allow_html=True)

    b1, b2, b3 = st.columns(3)
    b1.metric("Available Images", len(dataset_images))
    b2.metric("Saved Reports", len(all_saved_reports))
    b3.metric("Selected Model", active_model.split("/")[-1])

    if not dataset_images:
        st.warning(f"No images found in '{DATA_DIR}'. Place images there to enable batch analysis.")
    else:
        selected_filenames = st.multiselect(
            "Select images to investigate:",
            options=[p.name for p in dataset_images],
            default=[p.name for p in dataset_images],
        )

        with st.expander("🖼️ Preview Selected Images", expanded=False):
            preview_cols = st.columns(4)
            for idx, name in enumerate(selected_filenames):
                col = preview_cols[idx % 4]
                col.image(str(DATA_DIR / name), caption=name, use_container_width=True)

        can_batch = bool(selected_filenames) and bool(api_key_input) and bool(active_model)

        if st.button("Start Batch Processing", type="primary", disabled=not can_batch, use_container_width=True):
            progress_bar = st.progress(0)
            status_placeholder = st.empty()
            batch_results = []
            batch_errors = []

            total = len(selected_filenames)
            for i, name in enumerate(selected_filenames):
                status_placeholder.markdown(f"**Analyzing {i+1}/{total}:** `{name}`...")
                img_path = DATA_DIR / name
                try:
                    img_data = prepare_image(img_path)
                    res = investigate(
                        image_url=img_data,
                        filename=name,
                        model=active_model,
                        api_key=api_key_input,
                        enable_research=enable_research,
                    )
                    save_report(res)
                    batch_results.append(res)
                except Exception as ex:
                    batch_errors.append({"filename": name, "error": str(ex)})

                progress_bar.progress((i + 1) / total)

            status_placeholder.success("Batch processing complete!")
            st.session_state["last_batch_results"] = batch_results
            st.session_state["last_batch_errors"] = batch_errors

        last_batch = st.session_state.get("last_batch_results", [])
        last_errors = st.session_state.get("last_batch_errors", [])

        if last_batch or last_errors:
            st.divider()
            r1, r2 = st.columns(2)
            r1.metric("Completed Cases", len(last_batch))
            r2.metric("Failed Cases", len(last_errors))

            if last_errors:
                with st.expander("Review Errors", expanded=True):
                    st.json(last_errors)

            if last_batch:
                st.download_button(
                    "Download All Batch Reports (JSON)",
                    data=json.dumps(last_batch, indent=2, ensure_ascii=False),
                    file_name="batch_investigation_reports.json",
                    mime="application/json",
                )


# =========================================================
# ROUTE 4: REPORTS LIBRARY
# =========================================================

elif st.session_state["active_page"] == "Reports Library":
    st.markdown('<div class="page-heading">Reports Library</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-description">Search, review, and export all generated case files.</div>', unsafe_allow_html=True)

    reports = load_reports()
    if not reports:
        st.info("No saved reports found. Conduct an investigation to populate the library.")
    else:
        search_query = st.text_input("Filter reports by filename or landmark", placeholder="Search...")
        filtered = [
            (path, rep) for path, rep in reports
            if search_query.lower() in rep.get("filename", "").lower()
            or search_query.lower() in rep.get("report", "").lower()
        ]

        st.caption(f"Showing {len(filtered)} matching case files")

        for path, report_item in filtered:
            header_title = f"{report_item.get('filename', path.stem)}  ·  {report_item.get('created_at', '')}"
            with st.expander(header_title):
                c_img, c_rep = st.columns([0.4, 0.6])
                img_path = get_image_for_report(report_item)

                with c_img:
                    if img_path:
                        st.image(str(img_path), caption=report_item.get("filename"), use_container_width=True)
                    else:
                        st.caption("No thumbnail cached for this case.")

                with c_rep:
                    st.markdown(f"**Model:** `{report_item.get('model', 'Unknown')}`")
                    st.markdown(report_item.get("report", "")[:1200] + "...\n\n*(Full report expandable below)*")

                st.markdown(report_item.get("report", ""))

                dl1, dl2 = st.columns(2)
                with dl1:
                    st.download_button(
                        "Download JSON",
                        data=json.dumps(report_item, indent=2, ensure_ascii=False),
                        file_name=path.name,
                        mime="application/json",
                        key=f"dl_json_{path.name}",
                        use_container_width=True,
                    )
                with dl2:
                    st.download_button(
                        "Download Markdown",
                        data=report_item.get("report", ""),
                        file_name=f"{path.stem}.md",
                        mime="text/markdown",
                        key=f"dl_md_{path.name}",
                        use_container_width=True,
                    )


# =========================================================
# ROUTE 5: INVESTIGATION CHAT
# =========================================================

elif st.session_state["active_page"] == "Investigation Chat":
    st.markdown('<div class="page-heading">Investigation Follow-Up Chat</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-description">Ask detailed follow-up questions grounded strictly in the case report.</div>', unsafe_allow_html=True)

    reports = load_reports()

    if not reports:
        st.info("Analyze an image or create a report first to enable follow-up interrogation.")
    else:
        case_options = [
            f"{rep.get('filename', p.stem)} ({rep.get('created_at', 'Unknown Date')})"
            for p, rep in reports
        ]
        selected_idx = st.selectbox(
            "Select Investigation Case File",
            range(len(case_options)),
            format_func=lambda i: case_options[i],
        )

        active_path, active_case = reports[selected_idx]
        case_id = str(active_path.name)

        # Initialize per-case chat history
        if case_id not in st.session_state["chat_histories"]:
            st.session_state["chat_histories"][case_id] = []

        history = st.session_state["chat_histories"][case_id]

        chat_c1, chat_c2 = st.columns([0.3, 0.7])
        with chat_c1:
            img_path = get_image_for_report(active_case)
            if img_path:
                st.image(str(img_path), caption=active_case.get("filename"), use_container_width=True)
            st.caption(f"**Investigated:** {active_case.get('created_at')}")
            st.caption(f"**Model:** {active_case.get('model')}")

            if st.button("Clear Chat History", use_container_width=True):
                st.session_state["chat_histories"][case_id] = []
                st.rerun()

        with chat_c2:
            st.markdown(
                f"<div class='panel'><div class='eyebrow'>ACTIVE CASE TARGET</div>"
                f"<div class='panel-title'>{active_case.get('filename')}</div>"
                f"<div class='panel-description'>Answers are strictly grounded in the recorded visual observations and external findings.</div></div>",
                unsafe_allow_html=True,
            )

            # Display conversation
            for msg in history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            user_question = st.chat_input("Ask about architectural style, region, competing hypotheses, or uncertainties...")

            if user_question:
                if not api_key_input:
                    st.error("API key is required to query the model.")
                else:
                    # Append user message
                    history.append({"role": "user", "content": user_question})
                    with st.chat_message("user"):
                        st.markdown(user_question)

                    # Build system message grounded in report
                    system_prompt = (
                        "You are an objective forensic visual investigation assistant.\n"
                        "Answer questions about this monument strictly using the facts, observations, and "
                        "hypotheses documented in the case report below. Do NOT fabricate historical events, "
                        "locations, or verified sources that are not in the report. If the report does not "
                        "contain the answer, clearly state that it is unknown.\n\n"
                        f"INVESTIGATION CASE REPORT:\n{active_case.get('report', '')}"
                    )

                    prompt_messages = [{"role": "system", "content": system_prompt}]
                    # Keep last 8 turns for context
                    prompt_messages.extend(history[-8:])

                    with st.chat_message("assistant"):
                        with st.spinner("Analyzing case evidence..."):
                            try:
                                bot_reply = ask_model(
                                    prompt_messages,
                                    model=active_model,
                                    api_key=api_key_input,
                                    max_tokens=1000,
                                )
                                st.markdown(bot_reply)
                                history.append({"role": "assistant", "content": bot_reply})
                                st.session_state["chat_histories"][case_id] = history
                            except Exception as error:
                                st.error(f"Error generating answer: {error}")


# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="footer">
Forensic Visual Investigation AI &nbsp; | &nbsp;
OpenRouter Vision Multi-Model Architecture &nbsp; | &nbsp;
Evidence-first Methodology<br>
Visual identification reports represent analytical hypotheses. Cross-reference with primary archaeological and historical records.
</div>
""", unsafe_allow_html=True)