"""
OCR Text Parser module for Student Assignment Evaluator.
Extracts:
- Student ID (with OCR normalization like MCAOO1 / MCAOOI -> MCA001)
- Question ID (e.g. Question: Q1 -> Q1, with fallback semantic/fuzzy matching against questions.csv)
- Question Text (handles multi-line headers e.g. 'Question:' followed by question text)
- Student Answer (isolates between 'Student Answer:' and 'End of answer')
"""

import re
from typing import Dict, List, Optional
from rapidfuzz import fuzz
from src.data_loader import get_all_questions


def normalize_student_id(raw_id: str) -> str:
    """Normalize OCR artifacts in student ID (e.g. MCAOO1 / MCAOOI -> MCA001)."""
    clean = raw_id.upper().strip()
    m = re.search(r"MCA\s*([0-9OIl|]{1,4})", clean)
    if m:
        suffix = m.group(1).replace("O", "0").replace("I", "1").replace("L", "1").replace("|", "1")
        digits = re.sub(r"\D", "", suffix)
        if digits:
            return f"MCA{digits.zfill(3)}"
    return clean


def normalize_question_id(raw_qid: str) -> str:
    """Normalize question ID variations: 'Question: Q1', 'Q 1', 'Ql', '01' -> 'Q1'."""
    clean = raw_qid.upper().strip()
    m = re.search(r"\bQ\s*([1-8])\b", clean)
    if m:
        return f"Q{m.group(1)}"
    
    m = re.search(r"QUESTION\s*[:\-\s]*([1-8])\b", clean, flags=re.IGNORECASE)
    if m:
        return f"Q{m.group(1)}"
    
    m = re.search(r"\bQ\s*([LI|])\b", clean)
    if m:
        return "Q1"

    m = re.search(r"^Q?([1-8])\b", clean)
    if m:
        return f"Q{m.group(1)}"

    return ""


def recover_question_id_from_text(text: str) -> Optional[str]:
    """
    Fallback: Recover Question ID by matching extracted text against known questions
    using RapidFuzz partial token matching.
    """
    questions = get_all_questions()
    best_qid = None
    best_score = 0.0

    lower_text = text.lower()
    for qid, qdata in questions.items():
        q_text = qdata["question"].lower()
        score = fuzz.token_set_ratio(q_text, lower_text)
        if score > best_score:
            best_score = score
            best_qid = qid

    if best_score >= 50.0:
        return best_qid
    return None


def parse_ocr_text(lines: List[str], raw_text: Optional[str] = None) -> Dict[str, str]:
    """
    Parse OCR lines and raw text into structured fields:
    {
        'student_id': str,
        'question_id': str,
        'question_text': str,
        'student_answer': str,
        'raw_ocr_text': str
    }
    """
    if not lines and raw_text:
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]

    # Filter out header title lines like "MCA DEPARTMENT STUDENT ANSWER SHEET"
    clean_lines = []
    for l in lines:
        if re.search(r"MCA\s+DEPARTMENT|STUDENT\s+ANSWER\s+SHEET", l, flags=re.IGNORECASE):
            continue
        clean_lines.append(l)

    full_text = "\n".join(clean_lines) if clean_lines else (raw_text or "")
    
    student_id = ""
    question_id = ""
    question_text = ""
    student_answer = ""

    # Phase 1: Targeted Regex on full text
    # 1. Student ID pattern
    sid_match = re.search(
        r"Student\s*ID\s*[:\-\s]*([A-Za-z0-9|]+)", full_text, flags=re.IGNORECASE
    )
    if sid_match:
        student_id = normalize_student_id(sid_match.group(1))
    else:
        mca_match = re.search(r"\b(MCA[0-9OIl|]{1,4})\b", full_text, flags=re.IGNORECASE)
        if mca_match:
            student_id = normalize_student_id(mca_match.group(1))

    # 2. Question number / header pattern (e.g. Question: Q1 or Question 1)
    q_match = re.search(
        r"(?:Question|Q)\s*[:\-\s]*\s*(?:Q\s*)?([0-9LI|]+)\b",
        full_text,
        flags=re.IGNORECASE,
    )
    if q_match:
        raw_num = q_match.group(1)
        question_id = normalize_question_id(f"Q{raw_num}")

    # 3. Question Text extraction (search lines following 'Question:' marker)
    for idx, line in enumerate(clean_lines):
        clean_line = line.strip()
        if re.match(r"^Question\s*[:\-]?\s*$", clean_line, flags=re.IGNORECASE):
            if idx + 1 < len(clean_lines):
                candidate = clean_lines[idx + 1].strip()
                if not re.search(r"Student\s*Answer", candidate, flags=re.IGNORECASE):
                    question_text = candidate
                    break
        elif re.match(r"^Question\s*[:\-]?\s*(.+)$", clean_line, flags=re.IGNORECASE):
            m = re.match(r"^Question\s*[:\-]?\s*(.+)$", clean_line, flags=re.IGNORECASE)
            val = m.group(1).strip()
            if not re.match(r"^Q\s*[1-8]$", val, flags=re.IGNORECASE):
                question_text = val
                break

    # 4. Student Answer extraction
    # Match text following 'Student Answer:' strictly with a colon/semicolon/dash
    ans_match = re.search(
        r"Student\s*Answer\s*[:;\-\.]\s*([\s\S]*?)(?:End\s*of\s*answer|$)",
        full_text,
        flags=re.IGNORECASE,
    )
    if ans_match and ans_match.group(1).strip():
        student_answer = ans_match.group(1).strip()
    else:
        collecting = False
        ans_lines = []
        for line in clean_lines:
            if re.search(r"Student\s*Answer\s*[:;\-\.]", line, flags=re.IGNORECASE):
                collecting = True
                parts = re.split(r"Student\s*Answer\s*[:;\-\.]", line, flags=re.IGNORECASE)
                if len(parts) > 1 and parts[1].strip():
                    ans_lines.append(parts[1].strip())
                continue
            if collecting:
                if re.search(r"End\s*of\s*answer", line, flags=re.IGNORECASE):
                    break
                ans_lines.append(line.strip())
        if ans_lines:
            student_answer = " ".join(ans_lines).strip()

    # Clean student answer artifacts
    student_answer = re.sub(r"End\s*(?:of\s*)?answer.*$", "", student_answer, flags=re.IGNORECASE).strip()
    student_answer = re.sub(r"[:\s]+$", "", student_answer).strip()
    student_answer = re.sub(r"\s+", " ", student_answer).strip()

    # Phase 2: Fallback recovery for Question ID if missing or ambiguous
    if not question_id:
        recovered = recover_question_id_from_text(full_text)
        if recovered:
            question_id = recovered

    # If question text is empty but question_id is found, fill question from questions.csv
    if question_id:
        q_meta = get_all_questions().get(question_id)
        if q_meta and not question_text:
            question_text = q_meta["question"]

    return {
        "student_id": student_id,
        "question_id": question_id,
        "question_text": question_text,
        "student_answer": student_answer,
        "raw_ocr_text": full_text,
    }
