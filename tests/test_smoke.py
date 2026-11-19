import sys
import time
from pathlib import Path

proj_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(proj_dir))

from src.ocr_engine import extract_text_from_image, get_ocr_reader
from src.parser import parse_ocr_text

print("Pre-warming EasyOCR reader...")
t_warm = time.perf_counter()
get_ocr_reader(gpu=False)
print(f"EasyOCR reader warmed in {time.perf_counter() - t_warm:.2f}s")

samples = ["MCA001_Q1.png", "MCA002_Q4.png", "MCA003_Q8.png"]
img_dir = proj_dir / "data" / "answer_images"

for name in samples:
    p = img_dir / name
    print(f"\n--- Testing {name} ---")
    t0 = time.perf_counter()
    ocr_res = extract_text_from_image(p)
    t_ocr = time.perf_counter() - t0
    parsed = parse_ocr_text(ocr_res["lines"], ocr_res["raw_text"])
    t_total = time.perf_counter() - t0

    conf_pct = ocr_res["mean_confidence"] * 100.0
    print(f"Time: {t_total:.2f}s (OCR: {t_ocr:.2f}s, Conf: {conf_pct:.1f}%)")
    print(f"Extracted Student ID:  {parsed['student_id']}")
    print(f"Extracted Question ID: {parsed['question_id']}")
    print(f"Extracted Question:    {parsed['question_text']}")
    print(f"Extracted Answer:      {parsed['student_answer']}")
