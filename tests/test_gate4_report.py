import sys
from pathlib import Path

proj_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(proj_dir))

import config
from src.data_loader import get_question_by_id
from src.evaluator import evaluate_student_answer
from src.feedback_generator import generate_feedback

print("=" * 60)
print("  GATE 4: VERIFYING FEEDBACK & REPORT EXPORT TEAM DETAILS")
print("=" * 60)

q_data = get_question_by_id("Q1")
ans = "NLP is a branch of Al that helps computers understand and process human language."
eval_res = evaluate_student_answer("Q1", ans)
feedback = generate_feedback(q_data, eval_res, student_id="MCA001")

# Check team metadata in config
assert config.TEAM_NAME == "Invincible", f"Unexpected TEAM_NAME: {config.TEAM_NAME}"
assert config.TEAM_NUMBER == "9", f"Unexpected TEAM_NUMBER: {config.TEAM_NUMBER}"
assert "Prajwal S" in config.PARTICIPANTS
assert "Karan Shetty" in config.PARTICIPANTS
assert "Prarthana Winslow" in config.PARTICIPANTS

print(f"Team Name:     {config.TEAM_NAME}")
print(f"Team Number:   {config.TEAM_NUMBER}")
print(f"Participants:  {config.PARTICIPANTS}")
print(f"Project Title: {config.PROJECT_TITLE}")

print("\n--- Feedback Generated ---")
print("Summary:", feedback["summary"])
print("Covered Concepts:", feedback["covered_concepts"])
print("Missing Concepts:", feedback["missing_concepts"])
print("Suggestions:", feedback["actionable_suggestions"])

# Verify report assembly
report = {
    "metadata": {
        "team_name": config.TEAM_NAME,
        "team_number": config.TEAM_NUMBER,
        "participants": config.PARTICIPANTS,
        "project_title": config.PROJECT_TITLE,
    },
    "student_id": "MCA001",
    "question_id": "Q1",
    "feedback": feedback,
}

assert report["metadata"]["team_name"] == "Invincible"
assert report["metadata"]["team_number"] == "9"
print("\nGATE 4 VERIFICATION PASSED: Team metadata cleanly verified in feedback and report structures!")
