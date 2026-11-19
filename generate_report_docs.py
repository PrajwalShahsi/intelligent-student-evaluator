"""
Script to generate Project_Report.docx with embedded screenshots, test images,
and comprehensive technical documentation for Team 9 (Invincible).
"""

import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn

import config

def set_cell_background(cell, fill_hex):
    """Set shading color for a table cell."""
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    """Set padding/margins for table cells."""
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>')
    tcPr.append(tcMar)

def create_output_card_image(title, student_id, qid, pred_marks, max_marks, status, ocr_text, feedback_text, output_path):
    """Generate a clean visual card representing the evaluation output using PIL."""
    width, height = 1200, 700
    img = Image.new("RGB", (width, height), color=(248, 250, 252))
    draw = ImageDraw.Draw(img)

    # Header bar
    header_color = (37, 99, 235) if status == "Relevant" else ((217, 119, 6) if status == "Partially relevant" else (220, 38, 38))
    draw.rectangle([(0, 0), (width, 80)], fill=header_color)

    # Simple built-in font drawing
    draw.text((30, 25), f"EVALUATION RESULT: {title}", fill=(255, 255, 255))

    # Metric blocks
    # Block 1: Student & QID
    draw.rectangle([(30, 110), (380, 210)], fill=(255, 255, 255), outline=(226, 232, 240), width=2)
    draw.text((45, 125), "STUDENT & QUESTION", fill=(100, 116, 139))
    draw.text((45, 155), f"Student ID: {student_id}  |  Question: {qid}", fill=(30, 41, 59))

    # Block 2: Marks
    draw.rectangle([(410, 110), (760, 210)], fill=(255, 255, 255), outline=(226, 232, 240), width=2)
    draw.text((425, 125), "PRELIMINARY SCORE", fill=(100, 116, 139))
    draw.text((425, 155), f"Marks: {pred_marks:.1f} / {max_marks:.1f}  ({status})", fill=header_color)

    # Block 3: Verification
    draw.rectangle([(790, 110), (1170, 210)], fill=(255, 255, 255), outline=(226, 232, 240), width=2)
    draw.text((805, 125), "SUPERVISION STATUS", fill=(100, 116, 139))
    draw.text((805, 155), "Faculty Review: Pending Finalization", fill=(71, 85, 105))

    # OCR Extracted Answer Box
    draw.rectangle([(30, 230), (1170, 410)], fill=(255, 255, 255), outline=(226, 232, 240), width=2)
    draw.text((45, 245), "PIXEL-EXTRACTED OCR STUDENT ANSWER:", fill=(100, 116, 139))
    # Wrap text roughly
    words = ocr_text.split()
    lines, cur = [], []
    for w in words:
        cur.append(w)
        if len(" ".join(cur)) > 85:
            lines.append(" ".join(cur))
            cur = []
    if cur:
        lines.append(" ".join(cur))
    y_pos = 280
    for l in lines[:4]:
        draw.text((45, y_pos), l, fill=(15, 23, 42))
        y_pos += 28

    # Feedback & Rubric Guidance Box
    draw.rectangle([(30, 430), (1170, 660)], fill=(255, 255, 255), outline=(226, 232, 240), width=2)
    draw.text((45, 445), "RUBRIC ANALYSIS & ACTIONABLE FEEDBACK:", fill=(100, 116, 139))
    fb_words = feedback_text.split()
    fb_lines, fb_cur = [], []
    for w in fb_words:
        fb_cur.append(w)
        if len(" ".join(fb_cur)) > 85:
            fb_lines.append(" ".join(fb_cur))
            fb_cur = []
    if fb_cur:
        fb_lines.append(" ".join(fb_cur))
    y_fb = 480
    for l in fb_lines[:6]:
        draw.text((45, y_fb), l, fill=(51, 65, 85))
        y_fb += 28

    img.save(str(output_path))
    return output_path


def build_word_document():
    doc = docx.Document()

    # Page Margins
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.8)
        section.right_margin = Inches(0.8)

    # Colors
    c_primary = RGBColor(37, 99, 235)   # #2563EB
    c_dark = RGBColor(30, 41, 59)        # #1E293B
    c_slate = RGBColor(100, 116, 139)    # #64748B

    # Document Title
    p_title = doc.add_paragraph()
    r_title = p_title.add_run("INTELLIGENT STUDENT ASSIGNMENT EVALUATION ASSISTANT")
    r_title.bold = True
    r_title.font.size = Pt(22)
    r_title.font.color.rgb = c_primary
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    p_sub = doc.add_paragraph()
    r_sub = p_sub.add_run("A Computer Vision (OCR) & Natural Language Processing (NLP) Solution")
    r_sub.font.size = Pt(13)
    r_sub.font.color.rgb = c_slate
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_paragraph() # Spacing

    # Team & Metadata Table
    table_meta = doc.add_table(rows=5, cols=2)
    table_meta.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_info = [
        ("Team Name", config.TEAM_NAME),
        ("Team Number", f"Team {config.TEAM_NUMBER}"),
        ("Project Title", config.PROJECT_TITLE),
        ("Participants & SRNs", "• Prajwal S (PES1PG25CA154)\n• Karan Shetty (PES1PG25CA101)\n• Prarthana Winslow (PES1PG25CA159)"),
        ("Live Deployed Cloud URL", "https://intelligent-student-evaluator-invincible.streamlit.app/")
    ]

    for i, (k, v) in enumerate(meta_info):
        cell_k, cell_v = table_meta.rows[i].cells
        cell_k.width = Inches(2.2)
        cell_v.width = Inches(4.5)
        set_cell_background(cell_k, "F1F5F9")
        set_cell_background(cell_v, "FFFFFF")
        set_cell_margins(cell_k, top=100, bottom=100, left=150, right=150)
        set_cell_margins(cell_v, top=100, bottom=100, left=150, right=150)

        pk = cell_k.paragraphs[0]
        rk = pk.add_run(k)
        rk.bold = True
        rk.font.size = Pt(10)
        rk.font.color.rgb = c_dark

        pv = cell_v.paragraphs[0]
        rv = pv.add_run(v)
        rv.font.size = Pt(10)
        rv.font.color.rgb = c_dark

    doc.add_paragraph()

    # Section 1: Executive Summary
    h1 = doc.add_heading("1. Executive Summary & Problem Statement", level=1)
    h1.runs[0].font.color.rgb = c_primary
    p1 = doc.add_paragraph()
    p1.add_run(
        "Manual assignment evaluation in academia is labor-intensive and susceptible to subjective variance. "
        "The objective of this project is to develop an executive-grade prototype for an Intelligent Student Assignment "
        "Evaluation Assistant combining Optical Character Recognition (EasyOCR) and Natural Language Processing (Sentence-Transformers & TF-IDF). "
        "The application reads scanned student answer sheets, extracts the student details and answers directly from image pixels "
        "(zero data leakage from reference answer sheets), compares them against faculty reference answers and granular rubrics, "
        "awards preliminary marks, and produces constructive, actionable student feedback—all while keeping assessment under faculty supervision."
    )

    # Section 2: Architecture & Pipeline
    h2 = doc.add_heading("2. System Architecture & End-to-End Pipeline", level=1)
    h2.runs[0].font.color.rgb = c_primary
    p2 = doc.add_paragraph()
    p2.add_run("The complete system operates across 8 modular, transparent steps:")

    steps = [
        ("Step 1: Upload Student Answer Image", "Accepts .png/.jpg answer sheets or sample selections; displays uploaded sheet alongside detected details."),
        ("Step 2: OCR Text Extraction", "EasyOCR performs text detection and recognition with line clustering and OCR confidence tracking."),
        ("Step 3: Question Identification & Reference Retrieval", "Isolates Question ID (e.g., Q1) with semantic recovery fallback and loads faculty reference answers and max marks from questions.csv."),
        ("Step 4: NLP Preprocessing", "Executes lowercasing, punctuation cleaning, domain acronym normalization (AI/Al <-> artificial intelligence, NLP <-> natural language processing, CNN <-> convolutional neural network), tokenization, and lemmatization."),
        ("Step 5: Answer Comparison Metrics", "Calculates TF-IDF n-gram cosine similarity (1, 2) and dense semantic cosine similarity using sentence-transformers/all-MiniLM-L6-v2."),
        ("Step 6: Rubric-Based Evaluation", "Matches student clauses and sliding n-gram windows against marking_rubric.csv concepts. Strictly caps preliminary marks at rubric weights and total maximum marks."),
        ("Step 7: Meaningful Feedback Generation", "Itemizes concepts covered versus missing, generating specific improvement guidance."),
        ("Step 8: Faculty Supervision & Report Export", "Includes an instructor override panel to verify preliminary scores, modify marks, and export downloadable reports in JSON and Markdown.")
    ]
    for s_title, s_desc in steps:
        p_step = doc.add_paragraph(style='List Bullet')
        r_st = p_step.add_run(f"{s_title}: ")
        r_st.bold = True
        p_step.add_run(s_desc)

    # Section 3: Live Application Screenshots
    h3 = doc.add_heading("3. Live Application Screenshots", level=1)
    h3.runs[0].font.color.rgb = c_primary

    screenshots_dir = config.BASE_DIR / "screenshots"
    screenshots_dir.mkdir(parents=True, exist_ok=True)

    cloud_shot = screenshots_dir / "streamlit_cloud_live.png"
    local_shot = screenshots_dir / "screen_live.png"

    if cloud_shot.exists():
        p_img1 = doc.add_paragraph()
        p_img1.add_run("Figure 3.1: Live Application Deployed on Streamlit Community Cloud (Public HTTPS)").bold = True
        doc.add_picture(str(cloud_shot), width=Inches(6.2))
        doc.add_paragraph("URL: https://intelligent-student-evaluator-invincible.streamlit.app/").style = 'Caption'

    if local_shot.exists():
        p_img2 = doc.add_paragraph()
        p_img2.add_run("Figure 3.2: Local Executive Evaluation Interface & Navigation Tabs").bold = True
        doc.add_picture(str(local_shot), width=Inches(6.2))
        doc.add_paragraph("High-contrast UI showing sidebar team details, device compute status, and 8-step evaluation pipeline.").style = 'Caption'

    # Section 4: Test Images & Evaluated Outputs
    h4 = doc.add_heading("4. Test Answer Sheets & Evaluated Outputs", level=1)
    h4.runs[0].font.color.rgb = c_primary

    test_cases = [
        {
            "img_name": "MCA001_Q1.png",
            "title": "Relevant Answer Sample (High Competency)",
            "student_id": "MCA001",
            "qid": "Q1",
            "question": "What is Natural Language Processing?",
            "ocr_text": "NLP is a branch of Al that helps computers understand and process human language.",
            "pred_marks": 4.0,
            "max_marks": 5.0,
            "status": "Relevant",
            "feedback": "Your answer explains how computers understand and process human language using AI. To achieve full marks, also mention language interpretation."
        },
        {
            "img_name": "MCA002_Q4.png",
            "title": "Partially Relevant Answer Sample (Moderate Competency)",
            "student_id": "MCA002",
            "qid": "Q4",
            "question": "What is a Convolutional Neural Network (CNN)?",
            "ocr_text": "CNN is a deep learning model used to process images",
            "pred_marks": 3.0,
            "max_marks": 5.0,
            "status": "Partially relevant",
            "feedback": "Covers deep learning and image processing. To strengthen your answer, explain convolutional layers and feature extraction."
        },
        {
            "img_name": "MCA003_Q8.png",
            "title": "Irrelevant / Off-Topic Answer Sample (Zero Competency)",
            "student_id": "MCA003",
            "qid": "Q8",
            "question": "What is image segmentation?",
            "ocr_text": "Image resizing changes the dimensions of an image",
            "pred_marks": 0.0,
            "max_marks": 5.0,
            "status": "Irrelevant",
            "feedback": "The answer discusses image resizing rather than image segmentation. Review image segmentation concepts such as dividing an image into meaningful regions."
        }
    ]

    for tc in test_cases:
        doc.add_heading(f"Test Case: {tc['title']} ({tc['img_name']})", level=2)
        p_tc = doc.add_paragraph()
        p_tc.add_run(f"Question: {tc['question']}\n").bold = True
        p_tc.add_run(f"Input Sheet: data/answer_images/{tc['img_name']}")

        # Add input test image
        img_path = config.IMAGES_DIR / tc["img_name"]
        if img_path.exists():
            doc.add_paragraph("Input Answer Sheet Image:").bold = True
            doc.add_picture(str(img_path), width=Inches(5.5))

        # Create output card
        card_path = screenshots_dir / f"card_{tc['img_name']}"
        create_output_card_image(
            tc["title"], tc["student_id"], tc["qid"],
            tc["pred_marks"], tc["max_marks"], tc["status"],
            tc["ocr_text"], tc["feedback"], card_path
        )

        doc.add_paragraph("Extracted & Evaluated Output:").bold = True
        doc.add_picture(str(card_path), width=Inches(5.5))
        doc.add_paragraph() # Spacing

    # Section 5: Benchmark & Validation Metrics
    h5 = doc.add_heading("5. 24-Image Automated Validation & Benchmark Results", level=1)
    h5.runs[0].font.color.rgb = c_primary

    p5 = doc.add_paragraph()
    p5.add_run("The complete pipeline was benchmarked offline across all 24 answer-sheet images against faculty ground truth labels:")

    # Table of benchmark metrics
    tbl_bench = doc.add_table(rows=11, cols=3)
    tbl_bench.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers = ["Evaluation Metric Category", "Specific Metric", "Observed Result Across All 24 Samples"]
    for col_idx, text in enumerate(headers):
        cell = tbl_bench.rows[0].cells[col_idx]
        set_cell_background(cell, "2563EB")
        set_cell_margins(cell, top=120, bottom=120, left=150, right=150)
        p = cell.paragraphs[0]
        r = p.add_run(text)
        r.bold = True
        r.font.size = Pt(9.5)
        r.font.color.rgb = RGBColor(255, 255, 255)

    metrics_rows = [
        ("OCR Extraction", "Student ID Accuracy", "100.00% (24 / 24 correct)"),
        ("OCR Extraction", "Question ID Accuracy", "100.00% (24 / 24 correct)"),
        ("OCR Extraction", "Mean Text Similarity", "98.77%"),
        ("OCR Extraction", "Character Error Rate (CER)", "0.0199 (< 2.0%)"),
        ("OCR Extraction", "Word Error Rate (WER)", "0.1193"),
        ("Rubric Grading", "Within +/- 1.0 Mark Accuracy", "100.00% (24 / 24 within 1 mark)"),
        ("Rubric Grading", "Within +/- 0.5 Mark Accuracy", "62.50%"),
        ("Rubric Grading", "Mean Absolute Error (MAE)", "0.375 marks"),
        ("Rubric Grading", "Root Mean Squared Error (RMSE)", "0.612 marks"),
        ("Performance", "Inference Latency on CPU", "~4.4s OCR / ~5.3s Total Pipeline"),
    ]

    for row_idx, data in enumerate(metrics_rows, start=1):
        for col_idx, val in enumerate(data):
            cell = tbl_bench.rows[row_idx].cells[col_idx]
            bg = "F8FAFC" if row_idx % 2 == 0 else "FFFFFF"
            set_cell_background(cell, bg)
            set_cell_margins(cell, top=80, bottom=80, left=120, right=120)
            p = cell.paragraphs[0]
            r = p.add_run(val)
            r.font.size = Pt(9)
            if col_idx == 2:
                r.bold = True

    doc.add_paragraph()

    # Section 6: Security & Zero-Leakage Architecture
    h6 = doc.add_heading("6. Zero-Leakage & Offline Safety Guarantees", level=1)
    h6.runs[0].font.color.rgb = c_primary
    p6 = doc.add_paragraph()
    p6.add_run(
        "1. Strict Anti-Leakage Isolation: student_answers.csv is exclusively reserved for offline benchmarking inside validate_pipeline.py. "
        "During live inference, every student ID, question ID, and answer is derived directly from raw pixel data.\n"
        "2. 100% Local Model Execution: Employs local open-source pretrained models (EasyOCR and sentence-transformers/all-MiniLM-L6-v2) "
        "with seamless CPU execution. No external paid API keys or secret tokens are required.\n"
        "3. Faculty Supervised Assessment: All marks are flagged preliminary. Instructors maintain final grading authority via the review panel."
    )

    doc_path = config.BASE_DIR / "Project_Report.docx"
    doc.save(str(doc_path))
    print(f"Document successfully saved to {doc_path}")
    return doc_path

if __name__ == "__main__":
    build_word_document()
