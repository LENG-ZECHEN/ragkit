"""ragkit evaluation harness.

P2.5 modules:
  - ``schema``         — QAItem, JudgeScores, SweepResultRow, JSONL I/O
  - ``eval_retrieval`` — Recall@K / MRR / nDCG@10 / refusal_correct
  - ``run_grid``       — sweep orchestrator (concurrent, judge-free, ES mget)
  - ``judge_helper``   — human-in-the-loop read/write/merge
  - ``judge_prompts``  — Chinese rubric text, reused by the human judge

The earlier ``eval_judge`` module (LLM-as-judge via qwen-plus) was retired
in P2.5; the human (Opus 4.7 in the parent conversation) judges instead.
"""
