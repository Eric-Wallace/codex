
# CodeXGlue code-to-text docstring generation evaluation:

1. Generate the reference file by running `codexglue.py:make_codexglue_reference_file`. 
2. Evaluate model outputs:
``
python codexglue_bleu_evaluator.py path_to_gold_file < path_to_model_output_file 
```
