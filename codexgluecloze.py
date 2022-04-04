import sys
import argparse
import pprint
import pickle
from typing import List
import logging
import sys, json, os

import numpy as np
import tqdm
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
    parser.add_argument('--score_method', default='inf', help='"inf or lr or left mode')
    parser.add_argument('--output_dir', default='./evaluator/predictions/', help='directory to save output predictions')
    parser.add_argument('--cloze_path', default='/private/home/sida/extgit/CodeXGLUE/Code-Code/ClozeTesting-maxmin/')
    parser.add_argument('--leftpad', default=10, help='# tokens to pad the lhs of the infill pad', type=int)
    parser.add_argument('--rightpad', default=10, help='# tokens to pad the rhs of the infill', type=int)
    parser.add_argument('--breaktoken', action="store_true", help='break token on the infill')

def get_cloze_words(filename):
    with open(filename, 'r', encoding='utf-8') as fp:
        words = fp.read().split('\n')
    return words

def esl(text):
    return model._encode(text, strip_eos=True).tolist()


def expand(model, args, pre, suf, token):
    if args.breaktoken:
        return esl(pre), esl(suf), esl(token)
        
    ecomp = esl(pre + token + suf)
    epre = esl(pre)
    epretok = esl(pre + token)
    etoksuf = esl(token + suf)
    esuf= esl(suf)
    etok = esl(token)
    # assert(len(etok) == 1)
    itok = len(epretok) - 1
    
    eps = len(etok) + 1 
    for upper in range(itok-eps, len(ecomp)+1):
        if len(model._decode(ecomp[:upper])) >= len(pre + token):
            break
    
    for lower in range(itok+eps, -1, -1):
        if len(model._decode(ecomp[:lower])) <= len(pre):
            break
    lower = max(0, lower - args.leftpad)
    upper = min(len(ecomp), upper + args.rightpad)
    # print(lower, upper, model._decode(ecomp[lower:upper]))
    
    if not token in model._decode(ecomp[lower:upper]):
        print('tok', model._decode(ecomp[lower:upper]))
        raise Exception('shouldnt happen, didnt find anything containing the token...')
    return ecomp[:lower], ecomp[upper:], ecomp[lower:upper]


def build_seq(model, args, pre, suf, token):
    """
    find the largest span containing the given token and score it using the infilling model
    """
    pret, suft, tokt = expand(model, args, pre, suf, token)

    if args.score_method == 'inf':
        seq = pret + esl('<sentinel:0>') + suft + esl('<sentinel:1>') + esl('<sentinel:0>') + tokt + esl('<eoss>')
    elif args.score_method == 'lr':
        seq = pret + tokt + suft
    elif args.score_method == 'left':
        seq = pret + tokt
    # elif args.score_method == 'gen':
    #     seqg = pret + esl('<sentinel:0>') + suft + esl('<sentinel:1>') + esl('<sentinel:0>') 
    #     for f in model._generate(torch.tensor(seqg), 2, n=3, temperature=1, extra_encoded_stop_words=[esl('<eoss>')]):
    #         words, probs, flag = f
    #         print('pre\t', model._decode(pret))
    #         print('suf\t', model._decode(suft))
    #         print('tok\t', model._decode(tokt))
    #         print('g\t', model._decode(words))
    
    return seq

def cloze(model, args, pre, suf, options):
    mode = args.score_method
    if mode in ['codex', 'codexleft']:
        scores = []
        for tok in options:
            if mode == 'codex':
                scores.append(model.score_text([pre + tok + suf], scoring='sum')) 
            elif mode == 'codexleft':
                scores.append(model.score_text([pre + tok], scoring='sum')) 
    elif mode in ['inf', 'left', 'lr']:
        seqs = [torch.tensor(build_seq(model, args, pre, suf, w)) for w in options]
        scores = model.score_tokens(seqs)
    # elif mode == 'batch_inf':
    #      from priming_scorer import Scorer
    #      self = model
    #      pret, suft, tokt = expand(model, args, pre, suf, options[0])
    #      seq = pret + esl('<sentinel:0>') + suft + esl('<sentinel:1>') + esl('<sentinel:0>')
    #      seq = torch.tensor(seq).to(self.lm_model.device)
    #      print(len(seq), seq.size())
    #      decoder = Scorer(self.lm_model, min_len=len(seq), max_len=len(seq)+2, temperature=1.0, show_tqdm=False)
    #      decoder = decoder.to(self.lm_model.device)
    #      probs = decoder.score(seq).cpu()
    #      scores = []
    #      for tok in options:
    #          ind = model._encode(tok, strip_eos=True)
    #          # print(tok, ind, probs[ind])
    #          scores.append(probs[ind][0])
    maxind = np.argmax(scores)
    pred = options[maxind]
    return pred

def funtokenize(toks):
    """hacky tokenize to convert tokens back to python string that looks more like data"""
    out = ' '
    import keyword
    for t in toks:
        if len(t) == 0:
            out += ' '
        elif out[-1].isalnum() and t[0].isalnum():
            out += ' ' + t
        # elif t == ':':
        #     out += ':\n'
        # elif out[-1]==')' and t[0].isalnum():
        #     out += ' '
        #     out += t
        # elif t == ',':
        #     out += ', '
        else:
            out += t
            
    return out

def cloze_test(args, lang, model):
    cloze_words_file = os.path.join(args.cloze_path, './data', 'cloze-'+args.cloze_mode, 'cloze_test_words.txt')
    file_path = os.path.join(args.cloze_path, './data', 'cloze-'+args.cloze_mode, lang, 'clozeTest.json')

    lines = json.load(open(file_path))
    results = []
    words = get_cloze_words(cloze_words_file)
    for line in tqdm.tqdm(lines):
        # text = ' '.join(line['nl_tokens']) + ''.join(line['pl_tokens'])
        text = ' '.join(line['nl_tokens']) + funtokenize(line['pl_tokens'])
        # print('*' * 100)
        prefix, suffix = text.split('<mask>')
        MAX_LEN = 6000
        if len(text) > MAX_LEN:
            if len(prefix) > MAX_LEN/2:
                prefix = prefix[-MAX_LEN//2:]
            if len(suffix) > MAX_LEN/2:
                suffix = suffix[:MAX_LEN//2] 

        prefix, suffix = text.split('<mask>')
        pred = cloze(model, args, prefix, suffix, words)
        results.append({'idx': line['idx'], 'prediction': pred})


    dir = os.path.join(args.output_dir, lang)
    if not os.path.exists(dir):
        os.makedirs(dir)
    with open(os.path.join(dir, 'predictions.txt'), 'w', encoding='utf-8') as fp:
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


def print_results(args, lang):
    answers = read_answers(os.path.join(args.cloze_path, 'evaluator/answers/', lang, 'answers.txt'))
    predictions = read_predictions(os.path.join(args.output_dir, lang, 'predictions.txt'))
    acc = calculate_scores(answers, predictions)
    print('{}, {:.3f}'.format(lang, acc))


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
    langset = ['python', 'javascript', 'ruby', 'go',  'java', 'php']
    for lang in langset:
    # for lang in ['javascript']:
        cloze_test(args, lang, model)
        print_results(args, lang)
    
    for lang in langset:
        print_results(args, lang)
