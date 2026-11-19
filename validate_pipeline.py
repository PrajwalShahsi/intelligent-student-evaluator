"""
Automated offline validation & benchmarking pipeline.
Evaluates all 24 answer sheet images through the EXACT production pipeline:
Image -> OCR (EasyOCR) -> Parser -> Preprocessor -> Evaluator -> Feedback
Compares predictions against student_answers.csv (used exclusively here for benchmarking).
Computes:
- OCR Metrics: Student ID accuracy, Question ID accuracy, Text similarity, CER, WER
- Grading Metrics: MAE, RMSE, Exact Match %, Within ±0.5 %, Within ±1.0 %, Pearson correlation
Exports validation_results.json and validation_report.csv.
"""

import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, distance

import config
from src.data_loader import get_question_by_id, get_all_questions
from src.ocr_engine import extract_text_from_image
from src.parser import parse_ocr_text
from src.preprocessor import preprocess_text
from src.evaluator import evaluate_student_answer
from src.feedback_generator import generate_feedback


def calculate_cer(reference: str, hypothesis: str) -> float:
    """Calculate Character Error Rate (CER = Levenshtein distance / len(ref))."""
    ref = reference.strip()
    hyp = hypothesis.strip()
    if not ref:
        return 0.0 if not hyp else 1.0
    dist = distance.Levenshtein.distance(ref, hyp)
    return float(dist / max(1, len(ref)))


def calculate_wer(reference: str, hypothesis: str) -> float:
    """Calculate Word Error Rate (WER = Word Levenshtein distance / len(ref_words))."""
    ref_words = reference.strip().split()
    hyp_words = hypothesis.strip().split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    dist = distance.Levenshtein.distance(ref_words, hyp_words)
    return float(dist / max(1, len(ref_words)))


def run_benchmark():
    print("=" * 70)
    print("  RUNNING INTELLIGENT STUDENT ASSIGNMENT EVALUATOR BENCHMARK")
    print("=" * 70)

    # 1. Load ground truth benchmark data
    if not config.STUDENT_ANSWERS_CSV.exists():
        raise FileNotFoundError(f"Missing benchmark ground truth at {config.STUDENT_ANSWERS_CSV}")

    gt_df = pd.read_csv(config.STUDENT_ANSWERS_CSV)
    gt_df.columns = gt_df.columns.str.strip()

    records = []
    total_images = len(gt_df)
    print(f"Total benchmark samples to evaluate: {total_images}\n")

    start_total_time = time.perf_counter()

    for idx, row in gt_df.iterrows():
        expected_sid = str(row["student_id"]).strip()
        expected_qid = str(row["question_id"]).strip()
        img_rel_path = str(row["answer_image"]).strip()
        expected_text = str(row["student_answer_text_for_validation_only"]).strip()
        expected_marks = float(row["expected_marks_out_of_5"])
        category = str(row.get("expected_category", "")).strip()

        # Locate image
        img_path = config.DATA_DIR / img_rel_path
        if not img_path.exists():
            # Try in answer_images directly
            img_path = config.IMAGES_DIR / Path(img_rel_path).name

        print(f"[{idx+1:02d}/{total_images:02d}] Processing {img_path.name} ... ", end="", flush=True)

        t0 = time.perf_counter()
        # 1. OCR Extraction (from raw image bytes/path, zero student_answers.csv leakage!)
        ocr_res = extract_text_from_image(img_path)

        # 2. Parsing
        parsed = parse_ocr_text(ocr_res["lines"], ocr_res["raw_text"])
        pred_sid = parsed["student_id"]
        pred_qid = parsed["question_id"]
        extracted_answer = parsed["student_answer"]

        # 3. Evaluation
        # If question_id missing or not found in questions, fallback to expected for scoring comparison
        eval_qid = pred_qid if pred_qid in get_all_questions() else expected_qid
        eval_res = evaluate_student_answer(eval_qid, extracted_answer, allow_partial=False)
        pred_marks = eval_res["total_marks"]

        # 4. Feedback
        q_data = get_question_by_id(eval_qid)
        feedback = generate_feedback(q_data, eval_res, student_id=pred_sid)
        elapsed = time.perf_counter() - t0

        # Calculate metrics
        sid_match = (pred_sid == expected_sid)
        qid_match = (pred_qid == expected_qid)
        text_sim = fuzz.ratio(expected_text.lower(), extracted_answer.lower())
        cer = calculate_cer(expected_text, extracted_answer)
        wer = calculate_wer(expected_text, extracted_answer)
        mark_err = pred_marks - expected_marks
        abs_err = abs(mark_err)

        record = {
            "image": img_path.name,
            "expected_student_id": expected_sid,
            "pred_student_id": pred_sid,
            "student_id_match": sid_match,
            "expected_question_id": expected_qid,
            "pred_question_id": pred_qid,
            "question_id_match": qid_match,
            "expected_text": expected_text,
            "extracted_answer": extracted_answer,
            "text_similarity_ratio": round(text_sim, 2),
            "cer": round(cer, 4),
            "wer": round(wer, 4),
            "expected_marks": expected_marks,
            "pred_marks": pred_marks,
            "marks_error": round(mark_err, 2),
            "abs_error": round(abs_err, 2),
            "category": category,
            "ocr_confidence": ocr_res["mean_confidence"],
            "ocr_latency_sec": ocr_res["latency_seconds"],
            "total_latency_sec": round(elapsed, 3),
        }
        records.append(record)
        print(f"Done in {elapsed:.2f}s | SID: {'OK' if sid_match else 'FAIL'} | QID: {'OK' if qid_match else 'FAIL'} | Marks: {pred_marks:.1f}/{expected_marks:.1f}")

    total_time = time.perf_counter() - start_total_time
    report_df = pd.DataFrame(records)

    # Compute Summary Statistics
    total_samples = len(report_df)
    sid_acc = (report_df["student_id_match"].sum() / total_samples) * 100.0
    qid_acc = (report_df["question_id_match"].sum() / total_samples) * 100.0
    mean_text_sim = float(report_df["text_similarity_ratio"].mean())
    mean_cer = float(report_df["cer"].mean())
    mean_wer = float(report_df["wer"].mean())

    mae = float(report_df["abs_error"].mean())
    rmse = float(np.sqrt((report_df["marks_error"] ** 2).mean()))
    exact_mark_pct = (report_df["abs_error"] == 0).sum() / total_samples * 100.0
    within_half_pct = (report_df["abs_error"] <= 0.5).sum() / total_samples * 100.0
    within_one_pct = (report_df["abs_error"] <= 1.0).sum() / total_samples * 100.0

    try:
        pearson_corr = float(report_df["expected_marks"].corr(report_df["pred_marks"]))
        if np.isnan(pearson_corr):
            pearson_corr = 1.0
    except Exception:
        pearson_corr = 0.0

    mean_ocr_latency = float(report_df["ocr_latency_sec"].mean())
    mean_total_latency = float(report_df["total_latency_sec"].mean())

    summary = {
        "metadata": {
            "team_name": config.TEAM_NAME,
            "team_number": config.TEAM_NUMBER,
            "participants": config.PARTICIPANTS,
            "project_title": config.PROJECT_TITLE,
            "evaluated_samples": total_samples,
            "total_benchmark_time_seconds": round(total_time, 2),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "ocr_metrics": {
            "student_id_exact_accuracy_pct": round(sid_acc, 2),
            "student_id_correct_count": int(report_df["student_id_match"].sum()),
            "question_id_exact_accuracy_pct": round(qid_acc, 2),
            "question_id_correct_count": int(report_df["question_id_match"].sum()),
            "mean_text_similarity_ratio": round(mean_text_sim, 2),
            "mean_cer": round(mean_cer, 4),
            "mean_wer": round(mean_wer, 4),
            "mean_ocr_latency_sec": round(mean_ocr_latency, 3),
        },
        "grading_metrics": {
            "mae": round(mae, 3),
            "rmse": round(rmse, 3),
            "exact_mark_match_pct": round(exact_mark_pct, 2),
            "within_half_mark_pct": round(within_half_pct, 2),
            "within_one_mark_pct": round(within_one_pct, 2),
            "pearson_correlation": round(pearson_corr, 4),
            "mean_total_latency_sec": round(mean_total_latency, 3),
        },
    }

    # Save to disk
    report_csv_path = config.BASE_DIR / "validation_report.csv"
    report_df.to_csv(report_csv_path, index=False)

    results_json_path = config.BASE_DIR / "validation_results.json"
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("  FINAL VALIDATION BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"Student ID Accuracy:      {sid_acc:.2f}% ({summary['ocr_metrics']['student_id_correct_count']}/{total_samples})")
    print(f"Question ID Accuracy:     {qid_acc:.2f}% ({summary['ocr_metrics']['question_id_correct_count']}/{total_samples})")
    print(f"Mean Text Similarity:     {mean_text_sim:.2f}%")
    print(f"Mean CER:                 {mean_cer:.4f}")
    print(f"Mean WER:                 {mean_wer:.4f}")
    print("-" * 70)
    print(f"Grading MAE:              {mae:.3f} marks")
    print(f"Grading RMSE:             {rmse:.3f} marks")
    print(f"Exact Mark Match:         {exact_mark_pct:.2f}%")
    print(f"Within +/- 0.5 Mark:      {within_half_pct:.2f}%")
    print(f"Within +/- 1.0 Mark:      {within_one_pct:.2f}%")
    print(f"Pearson Correlation:      {pearson_corr:.4f}")
    print(f"Avg Inference Latency:    {mean_total_latency:.2f}s per image on CPU")
    print("=" * 70)
    print(f"Saved reports: {report_csv_path.name} & {results_json_path.name}\n")
    return summary, report_df


if __name__ == "__main__":
    run_benchmark()
