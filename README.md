
# TODO:
- [] clean up `causal_masking_infill` to take in a model path

# CodeXGlue code-to-text docstring generation:

1. Change the model path in `causal_masking_infill.py`.
2. Run `python codexglue.py` (with GPU). This runs the model on all 15k examples from the test set (but for diagnostic purposes it's probably better to just evaluate on a couple hundred to get a sense of BLEU).
3. Download the evaluator and gold docstrings from the CodeXGlue repo: https://github.com/microsoft/CodeXGLUE/tree/main/Code-Text/code-to-text

# Infilling evaluation (HumanEval)

1. Change the model path in `causal_masking_infill.py`.
2. Run `infill_evaluation.py:run_systematic_infill` with `eval_type="one_line"` (`all_lines` is untested).
3. Run `infill_evaluation.py:evaluate_one_line_systematic` to evaluate functional correctness of the infilled lines and report an average pass rate and exact match rate.
