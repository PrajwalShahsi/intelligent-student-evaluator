"""
Evaluation engine implementing Hybrid NLP (TF-IDF + Dense Semantic Embedding) + Rubric Matching.
Computes:
- TF-IDF Cosine Similarity (ngram_range=(1, 2))
- Dense Semantic Similarity (sentence-transformers/all-MiniLM-L6-v2)
- Multi-scale Clause-Level & Sliding-Window Rubric Concept Matching
- Best matching evidence extraction
- Dynamic preliminary marks calculation strictly capped at Max Marks
"""

import re
from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz

import config
from src.data_loader import get_rubric_for_question, get_question_by_id
from src.preprocessor import preprocess_text

_TRANSFORMER_MODEL = None


def get_embedding_model():
    """Lazily load SentenceTransformer model on CPU."""
    global _TRANSFORMER_MODEL
    if _TRANSFORMER_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _TRANSFORMER_MODEL = SentenceTransformer(
            config.EMBEDDING_MODEL_NAME,
            device="cpu",
        )
    return _TRANSFORMER_MODEL


def compute_tfidf_similarity(text1: str, text2: str) -> float:
    """Compute TF-IDF cosine similarity with unigrams and bigrams."""
    if not text1.strip() or not text2.strip():
        return 0.0
    try:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return float(np.clip(sim, 0.0, 1.0))
    except Exception:
        return 0.0


def compute_semantic_similarity(text1: str, text2: str) -> float:
    """Compute dense cosine similarity using SentenceTransformer."""
    if not text1.strip() or not text2.strip():
        return 0.0
    try:
        model = get_embedding_model()
        emb1 = model.encode([text1], convert_to_numpy=True)
        emb2 = model.encode([text2], convert_to_numpy=True)
        sim = cosine_similarity(emb1, emb2)[0][0]
        return float(np.clip(sim, 0.0, 1.0))
    except Exception:
        return 0.0


def generate_candidate_windows(text: str, max_n: int = 6) -> List[str]:
    """
    Generate sentences, clauses, and sliding word windows (n-grams) from text.
    Ensures short rubric concepts are matched against granular clauses rather than
    diluted across an entire multi-sentence paragraph.
    """
    if not text.strip():
        return []

    windows = set()
    # 1. Clause/Sentence splits
    clauses = re.split(r"[,;.\n]+|\b(?:and|that|which|to|by|for)\b", text, flags=re.IGNORECASE)
    for c in clauses:
        c_clean = c.strip()
        if len(c_clean.split()) >= 1:
            windows.add(c_clean)

    # 2. Sliding word windows (2 to max_n words)
    words = text.split()
    total_words = len(words)
    for n in range(2, min(max_n + 1, total_words + 1)):
        for i in range(total_words - n + 1):
            window = " ".join(words[i:i + n])
            windows.add(window)

    # Always include full text
    if text.strip():
        windows.add(text.strip())

    return list(windows)


def match_single_concept(
    concept_item: dict,
    student_answer_raw: str,
    student_answer_cleaned: str,
    candidate_windows: List[str],
    window_embeddings: Optional[np.ndarray] = None,
    allow_partial: bool = False,
) -> Dict[str, object]:
    """
    Match a single rubric concept against the student answer using:
    - Direct exact word-boundary match
    - RapidFuzz fuzzy partial match (with word boundary respect)
    - Clause-level / sliding window semantic similarity
    """
    alternatives = concept_item.get("alternatives", [concept_item.get("concept_raw", "")])
    concept_weight = concept_item.get("marks", 1.0)
    raw_concept = concept_item.get("concept_raw", "")

    best_score = 0.0
    best_evidence = ""
    match_method = "None"

    norm_student = student_answer_raw.lower()
    clean_student = student_answer_cleaned.lower()

    # Step A: Exact substring or lemmatized match with word boundaries
    for alt in alternatives:
        alt_clean = alt.lower().strip()
        if not alt_clean:
            continue

        # Check word boundary match (prevents "unlabelled" from matching "labelled")
        pattern = r"\b" + re.escape(alt_clean) + r"\b"
        if re.search(pattern, norm_student) or re.search(pattern, clean_student):
            best_score = 1.0
            best_evidence = alt
            match_method = "Exact Substring"
            break

        # RapidFuzz token match
        fuzz_token = fuzz.token_set_ratio(alt_clean, norm_student)
        fuzz_partial = fuzz.partial_ratio(alt_clean, norm_student)
        
        # Guard against antonym false positives (e.g. labelled vs unlabelled)
        if "un" + alt_clean in norm_student or "non" + alt_clean in norm_student:
            fuzz_token = 0.0
            fuzz_partial = 0.0

        max_fuzz = max(fuzz_token, fuzz_partial) / 100.0
        if max_fuzz > best_score:
            best_score = max_fuzz
            best_evidence = alt
            match_method = "Fuzzy Token Match"

    # Step B: Semantic Similarity with candidate windows if model available
    if best_score < 0.85 and candidate_windows:
        try:
            model = get_embedding_model()
            concept_embs = model.encode(alternatives, convert_to_numpy=True)
            if window_embeddings is not None and len(window_embeddings) > 0:
                sim_matrix = cosine_similarity(concept_embs, window_embeddings)
                max_idx = np.unravel_index(np.argmax(sim_matrix), sim_matrix.shape)
                sem_score = float(sim_matrix[max_idx])
                if sem_score > best_score:
                    best_score = sem_score
                    best_evidence = candidate_windows[max_idx[1]]
                    match_method = "Semantic Window Match"
        except Exception:
            pass

    # Determine status & marks
    status = "Missing"
    marks_awarded = 0.0

    if best_score >= config.CONCEPT_COVERED_THRESHOLD or best_score >= (config.FUZZY_RATIO_EXACT_THRESHOLD / 100.0):
        status = "Covered"
        marks_awarded = concept_weight
    elif allow_partial and (best_score >= config.CONCEPT_PARTIAL_THRESHOLD or best_score >= (config.FUZZY_RATIO_PARTIAL_THRESHOLD / 100.0)):
        status = "Partially Covered"
        marks_awarded = concept_weight * 0.5
    else:
        status = "Missing"
        marks_awarded = 0.0

    if status == "Missing":
        best_evidence = "—"

    return {
        "concept": raw_concept,
        "marks_weight": concept_weight,
        "score": round(best_score, 3),
        "status": status,
        "evidence": best_evidence,
        "method": match_method,
        "marks_awarded": marks_awarded,
    }


def evaluate_student_answer(
    question_id: str,
    student_answer_text: str,
    allow_partial: bool = False,
    relevance_threshold: float = 0.35,
) -> Dict[str, object]:
    """
    Main evaluation pipeline:
    1. Preprocessing of student answer and reference answer
    2. Comparison metrics: TF-IDF and Dense Semantic Cosine Similarity
    3. Relevance gating: Checks if answer is off-topic to question & reference
    4. Rubric concept evaluation across all points
    5. Aggregate preliminary marks calculation strictly capped at Max Marks
    """
    question_data = get_question_by_id(question_id)
    if not question_data:
        raise ValueError(f"Unknown Question ID: {question_id}")

    ref_answer = question_data["reference_answer"]
    question_title = question_data["question"]
    max_marks = question_data["maximum_marks"]

    # 1. NLP Preprocessing
    prep_student = preprocess_text(student_answer_text)
    prep_ref = preprocess_text(ref_answer)

    # 2. Similarity Metrics
    tfidf_sim = compute_tfidf_similarity(prep_student["cleaned_text"], prep_ref["cleaned_text"])
    semantic_sim = compute_semantic_similarity(student_answer_text, ref_answer)
    overall_sim = (config.TFIDF_WEIGHT * tfidf_sim) + (config.SEMANTIC_WEIGHT * semantic_sim)

    # Question topic semantic similarity
    q_sim = compute_semantic_similarity(student_answer_text, question_title)

    # Relevance Gating:
    # If the student answer is off-topic to the reference answer AND question title, gate to 0
    is_irrelevant = (
        (semantic_sim < relevance_threshold or q_sim < config.QUESTION_RELEVANCE_THRESHOLD)
        and (tfidf_sim < 0.20)
    )

    # 3. Rubric Evaluation
    rubrics = get_rubric_for_question(question_id)
    candidate_windows = generate_candidate_windows(prep_student["normalized_text"])

    window_embeddings = None
    if candidate_windows and not is_irrelevant:
        try:
            model = get_embedding_model()
            window_embeddings = model.encode(candidate_windows, convert_to_numpy=True)
        except Exception:
            window_embeddings = None

    rubric_breakdown = []
    total_awarded = 0.0

    for rubric_item in rubrics:
        if is_irrelevant:
            res = {
                "concept": rubric_item["concept_raw"],
                "marks_weight": rubric_item["marks"],
                "score": 0.0,
                "status": "Missing",
                "evidence": "—",
                "method": "Irrelevant Gate",
                "marks_awarded": 0.0,
            }
        else:
            res = match_single_concept(
                rubric_item,
                student_answer_raw=prep_student["normalized_text"],
                student_answer_cleaned=prep_student["cleaned_text"],
                candidate_windows=candidate_windows,
                window_embeddings=window_embeddings,
                allow_partial=allow_partial,
            )
        rubric_breakdown.append(res)
        total_awarded += res["marks_awarded"]

    # Strict capping at Max Marks
    final_marks = min(float(max_marks), float(total_awarded))
    coverage_pct = (sum(1 for r in rubric_breakdown if r["status"] == "Covered") / len(rubric_breakdown) * 100.0) if rubric_breakdown else 0.0

    return {
        "question_id": question_id,
        "question_text": question_title,
        "reference_answer": ref_answer,
        "max_marks": max_marks,
        "preprocessed_student_answer": prep_student,
        "preprocessed_reference_answer": prep_ref,
        "tfidf_similarity": round(tfidf_sim, 4),
        "semantic_similarity": round(semantic_sim, 4),
        "overall_similarity": round(overall_sim, 4),
        "question_similarity": round(q_sim, 4),
        "is_irrelevant": is_irrelevant,
        "concept_coverage_percent": round(coverage_pct, 1),
        "rubric_breakdown": rubric_breakdown,
        "total_marks": round(final_marks, 2),
    }
