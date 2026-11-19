"""
Data loader module for questions and marking rubrics.
STRICT RULE: Strictly NO loading of student_answers.csv in this module.
"""

from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import config


def load_questions(csv_path: Optional[Path] = None) -> pd.DataFrame:
    """Load questions.csv and return a normalized DataFrame."""
    path = csv_path or config.QUESTIONS_CSV
    if not path.exists():
        raise FileNotFoundError(f"Questions file not found at {path}")
    df = pd.read_csv(path)
    # Strip whitespace from string columns and column headers
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()
    df["maximum_marks"] = pd.to_numeric(df["maximum_marks"], errors="coerce").fillna(5).astype(int)
    return df


def get_all_questions() -> Dict[str, dict]:
    """Return all questions mapped by question_id."""
    df = load_questions()
    questions = {}
    for _, row in df.iterrows():
        qid = str(row["question_id"]).upper().strip()
        questions[qid] = {
            "question_id": qid,
            "subject": row.get("subject", ""),
            "question": row["question"],
            "maximum_marks": int(row["maximum_marks"]),
            "reference_answer": row["reference_answer"],
        }
    return questions


def get_question_by_id(question_id: str) -> Optional[dict]:
    """Retrieve question metadata and reference answer for a given question_id."""
    questions = get_all_questions()
    clean_id = question_id.upper().strip()
    return questions.get(clean_id)


def load_rubrics(csv_path: Optional[Path] = None) -> pd.DataFrame:
    """Load marking_rubric.csv and return a normalized DataFrame."""
    path = csv_path or config.RUBRIC_CSV
    if not path.exists():
        raise FileNotFoundError(f"Rubric file not found at {path}")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()
    df["marks"] = pd.to_numeric(df["marks"], errors="coerce").fillna(1.0).astype(float)
    return df


def get_rubric_for_question(question_id: str) -> List[dict]:
    """
    Retrieve list of expected concepts and their marks for a question.
    Parses alternative slash-separated concepts (e.g. 'Artificial Intelligence / AI')
    into convenient alternatives list for flexible evaluation.
    """
    df = load_rubrics()
    clean_id = question_id.upper().strip()
    sub_df = df[df["question_id"].str.upper() == clean_id]
    rubrics = []
    for _, row in sub_df.iterrows():
        raw_concept = str(row["concept_or_expected_point"]).strip()
        marks = float(row["marks"])
        # Split alternatives separated by '/'
        alternatives = [c.strip() for c in raw_concept.split("/") if c.strip()]
        rubrics.append({
            "concept_raw": raw_concept,
            "alternatives": alternatives,
            "marks": marks,
        })
    return rubrics
