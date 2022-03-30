import sys
import argparse
import pprint
import pickle
import tqdm
import logging
import sys, json, os
import numpy as np

import torch

# from datasets import load_dataset

from utils import build_docstring_infill_prompt, dump_git_status, dump_version_info
from models import make_model, Model, add_infilling_args, add_model_args, TruncationParameters


def make_parser():
    parser = argparse.ArgumentParser()
    add_model_args(parser)
    add_infilling_args(parser)

    parser.add_argument("--git_status", action="store_true")
    parser.add_argument("--resume", action="store_true")

    return parser

def add_cloze_args(parser):
    parser.add_argument('--cloze_mode', default='maxmin', help='"all" or "maxmin" mode')
    parser.add_argument('--score_method', default='inf', help='"inf or lr mode')
    parser.add_argument('--output_dir', default='./evaluator/predictions/', help='directory to save output predictions')
    parser.add_argument('--cloze_path', default='/private/home/sida/extgit/CodeXGLUE/Code-Code/ClozeTesting-maxmin/')
    

def get_cloze_words(filename):
    with open(filename, 'r', encoding='utf-8') as fp:
        words = fp.read().split('\n')
    return words

def score_token_infill(model, parts, token, top_p=1, max_tokens=2):
    self = model
    prompt = []
    # print(parts)
    # print('-'*100)
    for sentinel_ix, part in enumerate(parts):
        part_tokens = self._encode(part, strip_eos=True)
        prompt.extend(part_tokens.tolist())
        if sentinel_ix < len(parts) - 1:
            prompt.append(self.sentinel_id(sentinel_ix))
        else:
            # only makes sense to add an extra sentinel if we do have some text coming later, otherwise, we tend to just end the region immediately
            if self.extra_sentinel and len(part) > 0:
                prompt.append(self.sentinel_id(sentinel_ix))
    prompt.append(self.sentinel_id(0))
    prompt.extend(self._encode(token, strip_eos=True).tolist())
    prompt.append(self.EOSS_ID)
    
#     stop_words = self._extra_stop_words
#     encoded_stop_words = [self._encode(stop_word, strip_eos=True).tolist() for stop_word in stop_words]
#     encoded_stop_words.append([self.EOSS_ID])
#     with torch.no_grad():
#         for completion, scores in self._generate(
#             torch.tensor(prompt),
#             max_tokens=max_tokens,
#             top_p=top_p,
#             n=1,
#             temperature=0.1,
#             # extra_encoded_stop_words=encoded_stop_words,
#         ):
#             print(completion, scores)
#             print(model._decode(completion))

    # print(model._decode(torch.tensor(prompt)))
    # print(self.sentinel_id(0))
    return model.score_tokens([torch.tensor(prompt)])

def esl(text):
    return model._encode(text, strip_eos=True).tolist()

def score(model, args, pre, suf, token):
    """
    find the largest span containing the given token and score it using the infilling model
    """

    ecomp = esl(pre + token + suf)
    epre = esl(pre)
    epretok = esl(pre + token)
    etoksuf = esl(token + suf)
    esuf= esl(suf)
    etok = esl(token)
    assert(len(etok) == 1)
    itok = len(epretok) - 1
    
    eps = len(etok) + 1 
    for upper in range(itok-eps, len(ecomp)+1):
        if len(model._decode(ecomp[:upper])) >= len(pre + token):
            break
    
    for lower in range(itok+eps, -1, -1):
        if len(model._decode(ecomp[:lower])) <= len(pre):
            break
    
    if not token in model._decode(ecomp[lower:upper]):
        print(model._decode(ecomp[:lower]))
        print(upper, lower)
        print(len(token))
        print('tok', model._decode(ecomp[lower:upper]))
        print(model._decode(ecomp[upper:]))
        raise Exception('shouldnt happen, didnt find anything containing the token...')

    if args.score_method == 'inf':
        seq = ecomp[:lower] + esl('<sentinel:0>') + ecomp[upper:] + esl('<sentinel:1>') + esl('<sentinel:0>') + ecomp[lower:upper] + esl('<eoss>')
    elif args.score_method == 'lr':
        seq = ecomp[:lower] +  + ecomp[lower:upper] + ecomp[upper:]
    else:
        raise Exception('invalid score mode', args.score_method)
    # print(model._decode(seq))
    return model.score_tokens([torch.tensor(seq)])[0]   
    # print(model._decode(seq))


def funtokenize(toks):
    """hacky tokenize to convert tokens back to python string that looks more like data"""
    out = ' '
    for t in toks:
        if len(t) == 0:
            out += ' '
            continue
        elif out[-1].isalnum() and t[0].isalnum():
            out += ' ' + t
        elif t == ':':
            out += ':\n'
        elif out[-1]==')' and t[0].isalnum():
            out += '\n'
            out += t
        elif t == ',':
            out += ', '
        else:
            out += t
            
    return out

def cloze_test(args, lang, model):
    os.chdir(args.cloze_path)
    cloze_words_file = os.path.join('./data', 'cloze-'+args.cloze_mode, 'cloze_test_words.txt')
    file_path = os.path.join('./data', 'cloze-'+args.cloze_mode, lang, 'clozeTest.json')

    lines = json.load(open(file_path))
    results = []
    words = get_cloze_words(cloze_words_file)
    for line in tqdm.tqdm(lines):
        # text = ' '.join(line['nl_tokens']) + ''.join(line['pl_tokens'])
        text = ' '.join(line['nl_tokens']) + funtokenize(line['pl_tokens'])
        # print('*' * 100)
        if len(text) < 6000:
            # scores = model.score_text([text.replace('<mask>', w) for w in words])
            prefix, suffix = text.split('<mask>')
            scores = [score(model, args, [prefix, suffix], w) for w in words]
        else:
            scores = [0] * len(words)
            print(line['idx'], len(text))
        maxind = np.argmax(scores)
        pred = words[maxind]
        results.append({'idx': line['idx'], 'prediction': pred})

    with open(os.path.join(args.output_dir, lang, 'predictions.txt'), 'w', encoding='utf-8') as fp:
        for inst in results:
            fp.write(inst['idx']+'<CODESPLIT>'+inst['prediction']+'\n')
    print("ClozeMaxmin for {} finished".format(lang))
    return results


def read_answers(filename):
    answers = {}
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            answers[line.split('<CODESPLIT>')[0]] = line.split('<CODESPLIT>')[1]
    return answers


def read_predictions(filename):
    predictions = {}
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            predictions[line.split('<CODESPLIT>')[0]] = line.split('<CODESPLIT>')[1]
    return predictions


def calculate_scores(answers, predictions):
    scores = []
    for key in answers:
        if key not in predictions:
            logging.error("Missing prediction for index {}.".format(key))
            sys.exit()
        a = answers[key]
        p = predictions[key]
        scores.append(a==p)
    result = sum(scores) / len(scores)
    return result


def print_results(args):
    for lang in ['python', 'javascript', 'ruby', 'go',  'java', 'php']:
    # for lang in ['python', 'javascript']:
        answers = read_answers(os.path.join(args.cloze_path, 'evaluator/answers/', lang, 'answers.txt'))
        predictions = read_predictions(os.path.join(args.cloze_path, 'evaluator/predictions/', lang, 'predictions.txt'))
        acc = calculate_scores(answers, predictions)
        print('maxmin:{}\t {:.3f}'.format(lang, acc))


if __name__ == "__main__":
    print(' '.join(sys.argv))
    parser = make_parser()
    add_cloze_args(parser)
    
    args = parser.parse_args()
    pprint.pprint(vars(args))
    if args.git_status:
        dump_git_status()
        dump_version_info()

    model = make_model(args)
    for lang in ['python', 'javascript', 'ruby', 'go',  'java', 'php']:
    # for lang in ['ruby']:
        cloze_test(args, lang, model)

    print_results(args)
