"""
Intelligent Student Assignment Evaluation Assistant
Streamlit Web Application
Team: Invincible (Team 9)
Participants: Prajwal S(PES1PG25CA154), Karan Shetty(PES1PG25CA101), Prarthana Winslow(PES1PG25CA159)
"""

import io
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config
from src.data_loader import get_question_by_id, get_all_questions, get_rubric_for_question
from src.ocr_engine import extract_text_from_image, get_ocr_reader
from src.parser import parse_ocr_text
from src.preprocessor import preprocess_text
from src.evaluator import evaluate_student_answer, get_embedding_model
from src.feedback_generator import generate_feedback

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title=f"{config.PROJECT_TITLE} | Team {config.TEAM_NUMBER}",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CACHED MODEL LOADERS (Zero reload across interactions) ---
@st.cache_resource(show_spinner="Warming up local OCR Engine (EasyOCR)...")
def load_cached_ocr():
    return get_ocr_reader(gpu=False)

@st.cache_resource(show_spinner="Warming up Sentence Transformer (all-MiniLM-L6-v2)...")
def load_cached_embedder():
    return get_embedding_model()

# --- CUSTOM CSS STYLING ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .team-badge {
        display: inline-block;
        background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 100%);
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        margin-bottom: 10px;
    }
    .step-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 16px;
    }
    .step-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 10px;
    }
    .metric-box {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .disclaimer-banner {
        background-color: #FEF3C7;
        border-left: 4px solid #F59E0B;
        padding: 12px 16px;
        border-radius: 4px;
        color: #92400E;
        font-size: 0.9rem;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)


# --- SIDEBAR METADATA & SYSTEM STATUS ---
with st.sidebar:
    st.markdown(f'<div class="team-badge">Team {config.TEAM_NUMBER} • {config.TEAM_NAME}</div>', unsafe_allow_html=True)
    st.title("🎓 Assignment Assistant")
    st.markdown(f"**Project:** {config.PROJECT_TITLE}")
    st.divider()

    st.markdown("### 👥 Team Members")
    st.markdown("""
    - **Prajwal S** `(PES1PG25CA154)`
    - **Karan Shetty** `(PES1PG25CA101)`
    - **Prarthana Winslow** `(PES1PG25CA159)`
    """)
    st.divider()

    st.markdown("### ⚙️ System Status")
    import torch
    cuda_status = "CUDA GPU Available" if torch.cuda.is_available() else "Local CPU Mode"
    st.info(f"**Compute Device:** {cuda_status}\n\n**Embedding:** `all-MiniLM-L6-v2`\n\n**OCR:** `EasyOCR (English)`")

    st.markdown("### ⚠️ Notice")
    st.caption(config.DISCLAIMER)


# --- HEADER ---
col_head1, col_head2 = st.columns([4, 1])
with col_head1:
    st.markdown(f'<div class="main-header">🎓 {config.PROJECT_TITLE}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-header">AI-Assisted Student Evaluation with Optical Character Recognition & NLP Rubric Scoring</div>', unsafe_allow_html=True)
with col_head2:
    st.markdown(f"""
    <div style="text-align: right; padding-top: 10px;">
        <span style="background-color: #EFF6FF; color: #1E40AF; padding: 6px 12px; border-radius: 8px; font-weight: bold; border: 1px solid #BFDBFE;">
            Team {config.TEAM_NUMBER}: {config.TEAM_NAME}
        </span>
    </div>
    """, unsafe_allow_html=True)

st.markdown(f'<div class="disclaimer-banner">⚖️ <b>Faculty Supervision Policy:</b> {config.DISCLAIMER}</div>', unsafe_allow_html=True)

# Pre-warm cached models
load_cached_ocr()
load_cached_embedder()

# --- MAIN TABS ---
tab1, tab2, tab3 = st.tabs([
    "📝 Evaluate Answer Sheet",
    "📊 Validation & Benchmarks",
    "🏛️ Architecture & Spec"
])


# ==============================================================================
# TAB 1: EVALUATE ANSWER SHEET (All 8 Steps)
# ==============================================================================
with tab1:
    # State reset
    if "eval_reset_counter" not in st.session_state:
        st.session_state.eval_reset_counter = 0

    st.markdown("### Step 1: Upload Student Answer Image")
    st.caption("Upload a scanned answer sheet or select a benchmark sample image. The system strictly extracts all content from pixels without external data leakage.")

    col_u1, col_u2 = st.columns([1, 1])
    with col_u1:
        uploaded_file = st.file_uploader(
            "Choose an answer sheet image (.png, .jpg, .jpeg):",
            type=["png", "jpg", "jpeg"],
            key=f"file_uploader_{st.session_state.eval_reset_counter}"
        )
    with col_u2:
        # Sample image loader (reads ONLY raw .png files from data/answer_images)
        sample_images = sorted([p.name for p in config.IMAGES_DIR.glob("*.png")]) if config.IMAGES_DIR.exists() else []
        selected_sample = st.selectbox(
            "Or load a sample image from the dataset:",
            options=["-- Select Sample --"] + sample_images,
            key=f"sample_select_{st.session_state.eval_reset_counter}"
        )

    # Resolve active image
    active_image = None
    image_source_name = ""

    if uploaded_file is not None:
        active_image = Image.open(uploaded_file).convert("RGB")
        image_source_name = uploaded_file.name
    elif selected_sample and selected_sample != "-- Select Sample --":
        sample_path = config.IMAGES_DIR / selected_sample
        active_image = Image.open(sample_path).convert("RGB")
        image_source_name = selected_sample

    if active_image is not None:
        st.divider()
        col_img, col_proc = st.columns([1, 1])
        with col_img:
            st.image(active_image, caption=f"Active Answer Sheet: {image_source_name}", use_column_width=True)

        with col_proc:
            with st.spinner("Extracting text and isolating fields via EasyOCR..."):
                ocr_result = extract_text_from_image(active_image)
                parsed_data = parse_ocr_text(ocr_result["lines"], ocr_result["raw_text"])

            st.success(f"✅ OCR completed in {ocr_result['latency_seconds']:.2f}s | Confidence: {ocr_result['mean_confidence']*100:.1f}%")

            # Quick identified info pills
            st.markdown(f"""
            <div style="display: flex; gap: 10px; margin-bottom: 12px;">
                <div class="metric-box" style="flex: 1;">
                    <small>Detected Student ID</small><br>
                    <b style="font-size: 1.3rem; color: #2563EB;">{parsed_data['student_id'] or 'Not Found'}</b>
                </div>
                <div class="metric-box" style="flex: 1;">
                    <small>Detected Question ID</small><br>
                    <b style="font-size: 1.3rem; color: #059669;">{parsed_data['question_id'] or 'Not Found'}</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Step 2: OCR Text Extraction
        with st.expander("🔍 Step 2: OCR Extracted Text & Detailed Bounding Boxes", expanded=True):
            st.markdown(f"**Extracted Student ID:** `{parsed_data['student_id']}`")
            st.markdown(f"**Extracted Question ID:** `{parsed_data['question_id']}`")
            st.markdown(f"**Extracted Question:** {parsed_data['question_text']}")
            st.markdown(f"**Extracted Student Answer:**")
            st.info(parsed_data["student_answer"] or "No student answer text detected.")

            if st.checkbox("Show Raw OCR Bounding Boxes"):
                st.json(ocr_result["boxes"][:10])

        # Step 3: Question & Reference Retrieval
        matched_qid = parsed_data["question_id"]
        q_meta = get_question_by_id(matched_qid) if matched_qid else None

        if not q_meta:
            st.warning("⚠️ Question ID could not be matched automatically. Please select the question manually:")
            all_qids = list(get_all_questions().keys())
            manual_qid = st.selectbox("Select Matching Question:", options=all_qids)
            q_meta = get_question_by_id(manual_qid)
            matched_qid = manual_qid

        st.markdown("### Step 3: Question & Reference Retrieval")
        col_q1, col_q2 = st.columns([3, 1])
        with col_q1:
            st.markdown(f"**Subject:** `{q_meta['subject']}` | **Question ID:** `{q_meta['question_id']}`")
            st.markdown(f"**Question:** *{q_meta['question']}*")
            st.markdown(f"**Faculty Reference Answer:**")
            st.success(q_meta["reference_answer"])
        with col_q2:
            st.markdown(f"""
            <div class="metric-box">
                <small>Max Marks</small><br>
                <b style="font-size: 2rem; color: #1E293B;">{q_meta['maximum_marks']}</b>
            </div>
            """, unsafe_allow_html=True)

        # Step 4: NLP Preprocessing
        with st.expander("🧹 Step 4: NLP Preprocessing (Cleaned Tokens & Acronym Expansion)", expanded=False):
            prep_res = preprocess_text(parsed_data["student_answer"])
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                st.markdown("**Cleaned & Lemmatized Text:**")
                st.code(prep_res["cleaned_text"])
            with col_p2:
                st.markdown("**Extracted Key Concepts/Terms:**")
                st.write(prep_res["key_terms"])

        # Steps 5 & 6: Evaluation
        with st.spinner("Evaluating student answer against rubric & semantic reference..."):
            eval_result = evaluate_student_answer(
                question_id=matched_qid,
                student_answer_text=parsed_data["student_answer"],
                allow_partial=False
            )

        st.markdown("### Step 5: Answer Comparison Metrics")
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.markdown(f"""
            <div class="metric-box">
                <small>TF-IDF Similarity</small><br>
                <b style="font-size: 1.5rem; color: #2563EB;">{eval_result['tfidf_similarity']*100:.1f}%</b>
            </div>
            """, unsafe_allow_html=True)
        with kpi2:
            st.markdown(f"""
            <div class="metric-box">
                <small>Semantic Similarity</small><br>
                <b style="font-size: 1.5rem; color: #7C3AED;">{eval_result['semantic_similarity']*100:.1f}%</b>
            </div>
            """, unsafe_allow_html=True)
        with kpi3:
            st.markdown(f"""
            <div class="metric-box">
                <small>Rubric Coverage</small><br>
                <b style="font-size: 1.5rem; color: #059669;">{eval_result['concept_coverage_percent']:.1f}%</b>
            </div>
            """, unsafe_allow_html=True)
        with kpi4:
            st.markdown(f"""
            <div class="metric-box" style="background-color: #EFF6FF; border-color: #3B82F6;">
                <small>Preliminary Marks</small><br>
                <b style="font-size: 1.6rem; color: #1D4ED8;">{eval_result['total_marks']:.1f} / {eval_result['max_marks']}</b>
            </div>
            """, unsafe_allow_html=True)

        if eval_result.get("is_irrelevant"):
            st.error("🚨 **Off-Topic Answer Detected:** The response has low semantic relevance to the faculty question. Rubric points were not awarded.")

        # Step 6: Rubric Table
        st.markdown("### Step 6: Marking Rubric Evaluation Breakdown")
        rubric_data = []
        for r in eval_result["rubric_breakdown"]:
            status_icon = "✅ Covered" if r["status"] == "Covered" else ("⚠️ Partial" if r["status"] == "Partially Covered" else "❌ Missing")
            rubric_data.append({
                "Expected Rubric Concept": r["concept"],
                "Weight": f"{r['marks_weight']} mark",
                "Match Score": f"{r['score']*100:.1f}%",
                "Status": status_icon,
                "Evidence in Answer": r["evidence"],
                "Awarded Marks": f"{r['marks_awarded']:.1f}",
            })
        rubric_df = pd.DataFrame(rubric_data)
        st.dataframe(rubric_df, use_container_width=True, hide_index=True)

        # Step 7: Feedback
        st.markdown("### Step 7: Meaningful Feedback & Suggestions")
        feedback_data = generate_feedback(q_meta, eval_result, student_id=parsed_data["student_id"])

        col_fb1, col_fb2 = st.columns(2)
        with col_fb1:
            st.markdown("**✅ Concepts Covered:**")
            if feedback_data["covered_concepts"]:
                for c in feedback_data["covered_concepts"]:
                    st.markdown(f"- `{c}`")
            else:
                st.caption("No rubric concepts covered.")

        with col_fb2:
            st.markdown("**❌ Concepts Missing:**")
            if feedback_data["missing_concepts"]:
                for m in feedback_data["missing_concepts"]:
                    st.markdown(f"- `{m}`")
            else:
                st.caption("All rubric concepts covered.")

        st.markdown("**💡 Actionable Improvement Suggestions:**")
        st.info(feedback_data["feedback_text"])

        # Step 8: Faculty Supervision & Download
        st.markdown("### Step 8: Faculty Supervision & Evaluation Report")
        with st.container():
            st.markdown("#### 👨‍🏫 Faculty Review & Override Panel")
            col_rev1, col_rev2 = st.columns([1, 2])
            with col_rev1:
                faculty_marks = st.number_input(
                    "Final Verified Marks:",
                    min_value=0.0,
                    max_value=float(q_meta["maximum_marks"]),
                    value=float(eval_result["total_marks"]),
                    step=0.5,
                )
            with col_rev2:
                faculty_remarks = st.text_input(
                    "Faculty Notes / Justification:",
                    value="Marks verified by instructor.",
                )

            # Report Assembly
            eval_report = {
                "metadata": {
                    "team_name": config.TEAM_NAME,
                    "team_number": config.TEAM_NUMBER,
                    "participants": config.PARTICIPANTS,
                    "project_title": config.PROJECT_TITLE,
                    "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
                "student_info": {
                    "student_id": parsed_data["student_id"],
                    "question_id": matched_qid,
                    "subject": q_meta["subject"],
                    "image_name": image_source_name,
                },
                "question_details": {
                    "question": q_meta["question"],
                    "reference_answer": q_meta["reference_answer"],
                    "max_marks": q_meta["maximum_marks"],
                },
                "ocr_details": {
                    "mean_confidence": ocr_result["mean_confidence"],
                    "ocr_latency_sec": ocr_result["latency_seconds"],
                    "extracted_answer": parsed_data["student_answer"],
                },
                "evaluation_metrics": {
                    "tfidf_similarity": eval_result["tfidf_similarity"],
                    "semantic_similarity": eval_result["semantic_similarity"],
                    "overall_similarity": eval_result["overall_similarity"],
                    "preliminary_marks": eval_result["total_marks"],
                    "faculty_verified_marks": faculty_marks,
                    "faculty_remarks": faculty_remarks,
                },
                "rubric_breakdown": eval_result["rubric_breakdown"],
                "feedback": feedback_data,
            }

            col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
            with col_btn1:
                # JSON Download
                json_str = json.dumps(eval_report, indent=2)
                st.download_button(
                    label="📥 Download JSON Report",
                    data=json_str,
                    file_name=f"Evaluation_{parsed_data['student_id']}_{matched_qid}.json",
                    mime="application/json",
                    use_container_width=True,
                )

            with col_btn2:
                # Markdown / Text Report Download
                md_report = f"""# {config.PROJECT_TITLE}
**Team:** {config.TEAM_NAME} (Team #{config.TEAM_NUMBER})  
**Participants:** {config.PARTICIPANTS}  
**Timestamp:** {eval_report['metadata']['evaluation_timestamp']}  

---

### Student & Question Information
- **Student ID:** {parsed_data['student_id']}
- **Question ID:** {matched_qid} ({q_meta['subject']})
- **Question:** {q_meta['question']}
- **Reference Answer:** {q_meta['reference_answer']}
- **Maximum Marks:** {q_meta['maximum_marks']}

---

### OCR Extraction
- **Mean Confidence:** {ocr_result['mean_confidence']*100:.1f}%
- **Extracted Answer:** {parsed_data['student_answer']}

---

### Evaluation Results
- **Preliminary Marks:** {eval_result['total_marks']} / {q_meta['maximum_marks']}
- **Faculty Verified Marks:** {faculty_marks} / {q_meta['maximum_marks']}
- **Faculty Remarks:** {faculty_remarks}
- **TF-IDF Similarity:** {eval_result['tfidf_similarity']*100:.1f}%
- **Semantic Similarity:** {eval_result['semantic_similarity']*100:.1f}%

### Rubric Breakdown
| Concept | Status | Score | Marks Awarded | Evidence |
|---|---|---|---|---|
"""
                for r in eval_result["rubric_breakdown"]:
                    md_report += f"| {r['concept']} | {r['status']} | {r['score']*100:.1f}% | {r['marks_awarded']} | {r['evidence']} |\n"

                md_report += f"\n### Feedback & Suggestions\n{feedback_data['feedback_text']}\n"

                st.download_button(
                    label="📄 Download Markdown Report",
                    data=md_report,
                    file_name=f"Evaluation_{parsed_data['student_id']}_{matched_qid}.md",
                    mime="text/markdown",
                    use_container_width=True,
                )

            with col_btn3:
                if st.button("🔄 Evaluate Another Image", use_container_width=True):
                    st.session_state.eval_reset_counter += 1
                    st.rerun()

    else:
        st.info("👆 Please upload an answer sheet image or select a sample image above to begin evaluation.")


# ==============================================================================
# TAB 2: VALIDATION & BENCHMARKS
# ==============================================================================
with tab2:
    st.markdown("### 📊 Automated Benchmark & Validation Dashboard")
    st.caption("Displays offline validation across all 24 images in the dataset comparing OCR predictions and rubric grades against benchmark ground truth.")

    res_json_path = config.BASE_DIR / "validation_results.json"
    rep_csv_path = config.BASE_DIR / "validation_report.csv"

    col_bench_act, col_bench_info = st.columns([1, 3])
    with col_bench_act:
        run_bench = st.button("🚀 Run Full Pipeline Validation (24 Images)", use_container_width=True)

    if run_bench:
        with st.spinner("Executing full 24-image validation pipeline..."):
            from validate_pipeline import run_benchmark
            run_benchmark()
            st.success("Validation benchmark completed and metrics updated!")

    if res_json_path.exists() and rep_csv_path.exists():
        with open(res_json_path, "r", encoding="utf-8") as f:
            bench_results = json.load(f)
        bench_df = pd.read_csv(rep_csv_path)

        # KPI Metrics Cards
        st.markdown("#### OCR Extraction Benchmarks")
        o_col1, o_col2, o_col3, o_col4, o_col5 = st.columns(5)
        with o_col1:
            st.metric("Student ID Accuracy", f"{bench_results['ocr_metrics']['student_id_exact_accuracy_pct']:.1f}%")
        with o_col2:
            st.metric("Question ID Accuracy", f"{bench_results['ocr_metrics']['question_id_exact_accuracy_pct']:.1f}%")
        with o_col3:
            st.metric("Mean Text Similarity", f"{bench_results['ocr_metrics']['mean_text_similarity_ratio']:.1f}%")
        with o_col4:
            st.metric("Mean CER", f"{bench_results['ocr_metrics']['mean_cer']:.3f}")
        with o_col5:
            st.metric("Mean WER", f"{bench_results['ocr_metrics']['mean_wer']:.3f}")

        st.markdown("#### Grading & Scoring Accuracy Benchmarks")
        g_col1, g_col2, g_col3, g_col4, g_col5 = st.columns(5)
        with g_col1:
            st.metric("Grading MAE", f"{bench_results['grading_metrics']['mae']:.2f}")
        with g_col2:
            st.metric("Grading RMSE", f"{bench_results['grading_metrics']['rmse']:.2f}")
        with g_col3:
            st.metric("Exact Match %", f"{bench_results['grading_metrics']['exact_mark_match_pct']:.1f}%")
        with g_col4:
            st.metric("Within ±0.5 Mark", f"{bench_results['grading_metrics']['within_half_mark_pct']:.1f}%")
        with g_col5:
            st.metric("Pearson Correlation", f"{bench_results['grading_metrics']['pearson_correlation']:.3f}")

        st.divider()

        # Visualizations
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            st.markdown("##### Predicted Marks vs. Expected Marks")
            fig_scatter = px.scatter(
                bench_df,
                x="expected_marks",
                y="pred_marks",
                color="category",
                hover_data=["image", "expected_student_id", "expected_question_id"],
                title="Marks Correlation by Category",
                labels={"expected_marks": "Expected Marks (Ground Truth)", "pred_marks": "Predicted Marks"},
            )
            fig_scatter.add_trace(go.Scatter(x=[0, 5], y=[0, 5], mode="lines", name="Perfect Agreement", line=dict(dash="dash", color="gray")))
            st.plotly_chart(fig_scatter, use_container_width=True)

        with col_chart2:
            st.markdown("##### Error Distribution Across Images")
            fig_err = px.bar(
                bench_df,
                x="image",
                y="marks_error",
                color="abs_error",
                title="Per-Sample Marks Error (Predicted - Expected)",
                labels={"marks_error": "Marks Difference", "image": "Sample Image"},
            )
            st.plotly_chart(fig_err, use_container_width=True)

        st.markdown("#### Complete 24-Sample Validation Table")
        display_cols = [
            "image", "expected_student_id", "pred_student_id", "student_id_match",
            "expected_question_id", "pred_question_id", "question_id_match",
            "expected_marks", "pred_marks", "abs_error", "category", "text_similarity_ratio", "cer"
        ]
        st.dataframe(bench_df[display_cols], use_container_width=True, hide_index=True)
    else:
        st.info("Validation report has not been run yet. Click 'Run Full Pipeline Validation' above to execute.")


# ==============================================================================
# TAB 3: SYSTEM ARCHITECTURE & DOCUMENTATION
# ==============================================================================
with tab3:
    st.markdown("### 🏛️ System Architecture & Engineering Specifications")

    st.markdown("#### Complete End-to-End Pipeline Flow")
    st.code("""
[Student Answer Image] 
         ↓
  [EasyOCR Engine] (detail=1 bounding boxes, vertical clustering, horizontal sorting)
         ↓
  [Text Parser] (Student ID normalization, Question ID normalization & fallback recovery)
         ↓
  [Reference Retrieval] (Faculty Questions & Maximum Marks from questions.csv)
         ↓
  [NLP Preprocessing] (Tokenization, Acronym Expansion, Stopword Removal, Lemmatization)
         ↓
  [Answer Comparison] (TF-IDF N-Gram Cosine Similarity + all-MiniLM-L6-v2 Dense Similarity)
         ↓
  [Rubric Evaluator] (Clause-level & Sliding-Window Concept Matching, RapidFuzz Typos, Gating)
         ↓
  [Marks Calculation] (Capped at Rubric Weights & Max Marks)
         ↓
  [Feedback Engine] (Covered/Missing Concepts & Natural Actionable Suggestions)
         ↓
  [Faculty Supervision & Report Export] (Instructor override, JSON & Markdown export)
    """, language="text")

    st.markdown("#### Core Technical Principles")
    st.markdown("""
    1. **Zero Data Leakage:**
       - `student_answers.csv` is strictly isolated to offline benchmarking and the validation tab.
       - Every student answer is dynamically extracted from raw pixel input via EasyOCR.
    2. **100% Local & Offline Execution:**
       - Runs entirely without external paid LLM APIs or rate limits.
       - PyTorch runs in CPU mode with under ~4s per image latency.
       - Robust fallback stopwords and stemmers ensure zero crashes even when completely disconnected from the Internet.
    3. **Hybrid NLP + Rubric Engine:**
       - Rather than simple keyword matching, concepts are matched across sliding windows and clauses using Sentence-Transformers and fuzzy string alignment.
       - Irrelevant answers are gated to prevent isolated keyword false positives.
    4. **Faculty in the Loop:**
       - All marks are clearly labeled preliminary.
       - The system provides instant interactive overrides and notes for instructor accountability.
    """)

    st.markdown("#### Hackathon Team Information")
    st.markdown(f"""
    - **Team Name:** `{config.TEAM_NAME}`
    - **Team Number:** `{config.TEAM_NUMBER}`
    - **Participants:** `{config.PARTICIPANTS}`
    - **Project Title:** `{config.PROJECT_TITLE}`
    """)
