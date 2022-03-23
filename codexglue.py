import sys
import argparse
import pprint
import pickle
import tqdm

from datasets import load_dataset

from utils import build_docstring_infill_prompt, dump_git_status, dump_version_info
from models import make_model, Model, add_infilling_args, add_model_args, TruncationParameters

def run_codexglue_code_to_text(args, model: Model, result_base_path=None):
    data = load_dataset("code_x_glue_ct_code_to_text", "python", split="test")

    all_results = []
    problem_iterator = [{
        "id": d["id"],
        "prompt_parts": build_docstring_infill_prompt(d["original_string"], docstring_text=d["docstring"])
    } for d in data]

    if result_base_path is not None:
        result_txt_fname = f"{result_base_path}.txt"
        result_pkl_fname = f"{result_base_path}.pkl"
        response_pkl_fname = f"{result_base_path}_responses.pkl"
    else:
        result_txt_fname = "codexglue_code_to_text.txt"
        result_pkl_fname = "codexglue_code_to_text.pkl"
        response_pkl_fname = "codexglue_code_to_text_responses.pkl"
    if args.resume:
        start = sum(1 for line in open(result_txt_fname))
        print(f"==== Resuming from line {start} of {result_txt_fname} ====")
        result_txt = open(result_txt_fname, "a")
    else:
        start = 0
        print(f"==== Writing results to new file {result_txt_fname} ====")
        result_txt = open(result_txt_fname, "w")

    all_results = []

    responses = {}

    with tqdm.tqdm(problem_iterator, ncols=120) as pbar:
        for i, problem in enumerate(pbar):
            if i < start:
                continue
            try:
                prompt_parts = problem["prompt_parts"]
                truncation_parameters = [
                        TruncationParameters.from_heuristics(args.truncation_heuristics)
                    ]
                kwargs = dict(
                    verbose=False, n=args.num_candidates,
                    bidirectional_generation=args.bidirectional_generation, bidirectional_scoring=args.bidirectional_scoring,
                    truncation_parameters=truncation_parameters,
                    scoring=args.candidate_scoring,
                    stop_words=['"""', '    """'] # these are both their own tokens
                )
                if args.max_tokens is not None:
                    kwargs['max_tokens'] = args.max_tokens
                kwargs.update(sampling=True, top_p=args.top_p, temperature=args.temperature, beam=args.beam)
                sorted_choices, response = model.rank_infills(prompt_parts, **kwargs)
                responses[i] = response
                top_choice = sorted_choices[0]

                infill_result = problem.copy()
                infill_result["text"] = top_choice["infills"][0]
                infill_result["complete"] = top_choice["complete"]
                infill_result["text_untruncated"] = top_choice["infills_untruncated"][0]

                pbar.set_postfix({"output": infill_result["text"]})

                all_results.append(infill_result)

                # Write docstring to file for BLEU eval
                result_txt.write(f"{i}\t{infill_result['text']}\n")
                if i % 50 == 0:
                    result_txt.flush()
            except Exception as e:
                print(f"Error on {i}")
                print(e)
                import pdb; pdb.set_trace()

    # Note: since we don't save until the end of the run, these results are
    #  incomplete if we're resuming
    with open(result_pkl_fname, "wb") as f:
        pickle.dump(all_results, f)

    with open(responses, "wb") as f:
        pickle.dump(responses, f)

    result_txt.close()

def make_parser():
    parser = argparse.ArgumentParser()
    add_model_args(parser)
    add_infilling_args(parser)

    parser.add_argument("--result_base_path")
    parser.add_argument("--git_status", action="store_true")
    parser.add_argument("--resume", action="store_true")

    return parser


if __name__ == "__main__":
    print(' '.join(sys.argv))
    parser = make_parser()
    args = parser.parse_args()
    pprint.pprint(vars(args))
    if args.git_status:
        dump_git_status()
        dump_version_info()

    model = make_model(args)
    run_codexglue_code_to_text(args, model, result_base_path=args.result_base_path)
