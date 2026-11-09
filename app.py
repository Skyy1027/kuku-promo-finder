import json
import streamlit as st
import promo_finder as pf

st.set_page_config(page_title="Promo Finder", layout="wide")
st.title("Promo Finder")
st.caption(
    "Upload timestamped microdrama screenplays (one episode per file, or a merged file). "
    "Files are processed in memory. Optional model responses are cached in this app process by a SHA-256 prompt key."
)

with st.sidebar:
    st.header("Selection settings")
    top = st.number_input("Maximum promos per episode", min_value=1, max_value=2, value=2, step=1)
    early = st.number_input("Early-episode prior (episodes 1-3)", min_value=0.0, max_value=3.0, value=0.5, step=0.1)
    stitch = st.number_input("Two-part cut penalty", min_value=0.0, max_value=5.0, value=1.5, step=0.1)
    min_score = st.number_input("Minimum normalized score (0-10)", min_value=0.0, max_value=10.0, value=0.0, step=0.5)
    st.divider()
    scorer = st.selectbox("Scorer", ["Rules only (free)", "OpenAI-compatible API", "Anthropic API"])
    llm = "none" if scorer.startswith("Rules") else ("openai" if scorer.startswith("OpenAI") else "anthropic")
    llm_config = None
    llm_ready = True
    workers = 4
    if llm != "none":
        api_key = st.text_input("API key", type="password", help="Used only for requests from this Streamlit session; not written to environment variables.")
        model = st.text_input("Model ID", value="", placeholder="Enter a model ID supported by your provider")
        default_base = "https://api.openai.com/v1" if llm == "openai" else "https://api.anthropic.com"
        base_url = st.text_input("Base URL", value=default_base)
        workers = st.slider("Concurrent episode calls", min_value=1, max_value=8, value=4)
        llm_config = {"api_key": api_key, "model": model.strip(), "base_url": base_url.strip()}
        llm_ready = bool(api_key.strip() and model.strip())
        if not llm_ready:
            st.sidebar.warning("Enter both an API key and a model ID to enable LLM scoring.")

files = st.file_uploader(
    "Screenplay files (.txt / .md)", type=["txt", "md"], accept_multiple_files=True
)

if st.button("Find promos", disabled=(not files or not llm_ready), type="primary"):
    episodes = []
    for uploaded in files:
        try:
            content = uploaded.getvalue().decode("utf-8", errors="replace")
            found = pf.load_text(content, uploaded.name)
            accepted = [ep for ep in found if ep["events"]]
            episodes.extend(accepted)
            if len(accepted) != len(found):
                st.warning(f"{uploaded.name}: one or more sections had no timestamped events and were skipped.")
        except Exception as exc:
            st.error(f"Could not parse {uploaded.name}: {exc}")
    if not episodes:
        st.error("No timestamped episodes were found. Check the input format.")
    else:
        with st.spinner(f"Scoring {len(episodes)} episode(s)..."):
            try:
                promos, meta = pf.run_pipeline(
                    episodes, int(top), float(early), float(stitch), llm,
                    float(min_score), llm_config=llm_config, llm_workers=int(workers), return_meta=True
                )
            except Exception as exc:
                st.exception(exc)
                promos, meta = [], {}
        if meta:
            st.success(f"{len(promos)} promos from {meta['episodes']} episode(s)")
            counts = meta.get("llm_status_counts", {})
            if counts.get("fallback"):
                st.warning(f"LLM failed for {counts['fallback']} episode(s); rule-based candidates were used for those episodes.")
            if counts.get("all_rejected"):
                st.info(f"The LLM rejected all shortlisted candidates in {counts['all_rejected']} episode(s); those episodes have no output promo.")
            for warning in meta.get("warnings", []):
                with st.expander(f"Episode {warning['episode']} - {warning['status']}"):
                    st.write(warning["message"])
        if promos:
            rows = [{
                "rank": i + 1,
                "tier": "A (review first)" if i < 10 else "B (backup)" if i < 30 else "C (long tail)",
                "episode": p["episode"],
                "segments": " + ".join(f"{pf.mmss(a)}-{pf.mmss(b)}" for a, b in p["segs"]),
                "seconds": round(p["duration"], 1),
                "score / 10": p["score"],
                "hook": p["hook"],
                "cliffhanger": p["cliff"],
                "why": p.get("why", pf.explain(p)),
                "scorer": p.get("scorer", "rules"),
            } for i, p in enumerate(promos)]
            st.dataframe(rows, use_container_width=True, hide_index=True)
            import tempfile
            with tempfile.TemporaryDirectory() as directory:
                pf.write_outputs(promos, directory, len(episodes))
                with open(f"{directory}/promos.json", encoding="utf-8") as f:
                    json_text = f.read()
                with open(f"{directory}/promos.md", encoding="utf-8") as f:
                    markdown_text = f.read()
            c1, c2 = st.columns(2)
            c1.download_button("Download promos.json", json_text, "promos.json", "application/json", use_container_width=True)
            c2.download_button("Download promos.md", markdown_text, "promos.md", "text/markdown", use_container_width=True)
        else:
            st.info("No candidates passed the current score and standalone filters. Lower the score threshold or review the model's decisions.")
