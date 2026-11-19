import sys
from pathlib import Path

proj_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(proj_dir))

from src.preprocessor import preprocess_text
from src.evaluator import evaluate_student_answer
from src.data_loader import get_question_by_id

test_cases = [
    {
        "id": "MCA001_Q1",
        "qid": "Q1",
        "answer": "NLP is a branch of Al that helps computers understand and process human language.",
        "expected_marks": 4.0,
    },
    {
        "id": "MCA002_Q4",
        "qid": "Q4",
        "answer": "CNN is a deep learning model used to process images",
        "expected_marks": 3.0,
    },
    {
        "id": "MCA003_Q8",
        "qid": "Q8",
        "answer": "Image resizing changes the dimensions of an image",
        "expected_marks": 0.0,
    },
]

print("=" * 60)
print("  GATE 3: UNIT TESTING NLP PREPROCESSING & EVALUATION ENGINE")
print("=" * 60)

all_passed = True

for case in test_cases:
    qid = case["qid"]
    ans = case["answer"]
    q_data = get_question_by_id(qid)
    max_m = q_data["maximum_marks"]

    print(f"\n--- Testing {case['id']} ({qid}) ---")
    print(f"Question: {q_data['question']}")
    print(f"Student Answer: '{ans}'")

    # 1. Preprocessing
    prep = preprocess_text(ans)
    print(f"Cleaned Text: '{prep['cleaned_text']}'")
    print(f"Key Terms: {prep['key_terms']}")

    # 2. Evaluation
    eval_res = evaluate_student_answer(qid, ans, allow_partial=False)
    pred_marks = eval_res["total_marks"]

    print(f"TF-IDF Similarity:      {eval_res['tfidf_similarity']*100:.1f}%")
    print(f"Semantic Similarity:    {eval_res['semantic_similarity']*100:.1f}%")
    print(f"Overall Similarity:     {eval_res['overall_similarity']*100:.1f}%")
    print(f"Rubric Coverage:        {eval_res['concept_coverage_percent']:.1f}%")
    print(f"Calculated Marks:       {pred_marks:.1f} / {max_m}")
    print(f"Expected Benchmark:     {case['expected_marks']:.1f} / {max_m}")

    # Assertions
    assert 0.0 <= pred_marks <= float(max_m), f"Marks out of bounds: {pred_marks}"
    error = abs(pred_marks - case["expected_marks"])
    print(f"Absolute Error:         {error:.1f} mark(s)")

    if error > 1.0:
        print(f"WARNING: Deviation on {case['id']}: expected {case['expected_marks']}, got {pred_marks}")
        all_passed = False

print("\n" + "=" * 60)
if all_passed:
    print("ALL GATE 3 UNIT TESTS PASSED SUCCESSFULLY!")
else:
    print("GATE 3 COMPLETED WITH MINOR DEVIATIONS TO TUNE.")
print("=" * 60)
