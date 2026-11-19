"""
Feedback generation module for Student Assignment Evaluator.
Implements Step 7 without external APIs:
- Summarizes concepts covered and missing
- Generates actionable, student-specific improvement suggestions
"""

from typing import Dict, List


def generate_feedback(
    question_data: dict,
    eval_results: dict,
    student_id: str = "Student",
) -> Dict[str, object]:
    """
    Generate constructive, natural feedback based on rubric evaluation results.
    Returns:
    {
        'summary': str,
        'covered_concepts': List[str],
        'missing_concepts': List[str],
        'partial_concepts': List[str],
        'actionable_suggestions': List[str],
        'feedback_text': str
    }
    """
    rubric_breakdown = eval_results.get("rubric_breakdown", [])
    total_awarded = eval_results.get("total_marks", 0.0)
    max_marks = eval_results.get("max_marks", 5.0)
    semantic_sim = eval_results.get("semantic_similarity", 0.0)

    covered = []
    missing = []
    partial = []

    for item in rubric_breakdown:
        concept = item["concept"]
        status = item["status"]
        if status == "Covered":
            covered.append(concept)
        elif status == "Partially Covered":
            partial.append(concept)
        else:
            missing.append(concept)

    suggestions = []

    # Check for complete irrelevance or very low relevance
    if total_awarded == 0 and semantic_sim < 0.40:
        summary = "The submitted answer appears off-topic or irrelevant to the question asked."
        suggestions.append(
            f"Review the core subject matter for '{question_data.get('question', 'this question')}'. "
            f"Ensure your response directly addresses key concepts like: {', '.join(missing[:3])}."
        )
    elif total_awarded >= max_marks:
        summary = "Excellent answer! You have comprehensively covered all key rubric concepts."
        suggestions.append("Keep up the high standard of clarity, technical precision, and complete coverage.")
    else:
        summary = f"Good effort. Your answer covers {len(covered)} of {len(rubric_breakdown)} expected concept(s)."
        if missing:
            missing_str = ", ".join(f"'{m}'" for m in missing)
            suggestions.append(
                f"To achieve full marks, expand your explanation to include: {missing_str}."
            )
        if partial:
            partial_str = ", ".join(f"'{p}'" for p in partial)
            suggestions.append(
                f"Elaborate further on: {partial_str} with specific technical details."
            )

    feedback_text = summary
    if suggestions:
        feedback_text += "\n\n**Suggestions for Improvement:**\n" + "\n".join(f"- {s}" for s in suggestions)

    return {
        "summary": summary,
        "covered_concepts": covered,
        "missing_concepts": missing,
        "partial_concepts": partial,
        "actionable_suggestions": suggestions,
        "feedback_text": feedback_text,
    }
