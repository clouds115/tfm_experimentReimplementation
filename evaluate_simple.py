# NOTE: TFM Modification: This script has been modified in order to avoid RQ processing and implement PREDS vs. GOLDS comparison and evaluation
# with the four key metrics utilized in the original paper. This modification has been performed so that the evaluations performed utilize the same
# metric code as in the original paper.

import re
import sys
import numpy as np
import argparse

sys.path.append("metric")
from metric.smooth_bleu import codenn_smooth_bleu
from metric.meteor.meteor import Meteor
from metric.rouge.rouge import Rouge
from metric.cider.cider import Cider


def read_files(gold_path, pred_path):
    with open(gold_path, 'r', encoding="utf-8") as f:
        refs = [line.strip().lower() for line in f]

    with open(pred_path, 'r', encoding="utf-8") as f:
        preds = [line.strip().lower() for line in f]

    assert len(refs) == len(preds), "Mismatch between gold and predictions!"

    refs = [[re.split(r'(\W)', r) for r in refs]]
    preds = [re.split(r'(\W)', p) for p in preds]

    return refs[0], preds


def compute_bleu(refs, preds):
    r_str_list = []
    p_str_list = []

    for r, p in zip(refs, preds):
        r_str_list.append([" ".join(r)])
        p_str_list.append(" ".join(p))

    bleu_list, _ = codenn_smooth_bleu(r_str_list, p_str_list)
    print("BLEU:", round(bleu_list[0], 2))
    return round(bleu_list[0], 2)


def compute_other_metrics(refs, preds):
    refs_dict = {}
    preds_dict = {}

    for i in range(len(preds)):
        preds_dict[i] = [" ".join(preds[i])]
        refs_dict[i] = [" ".join(refs[i])]

    meteor_score, meteor_scores = Meteor().compute_score(refs_dict, preds_dict)
    meteor_score = np.mean(meteor_scores)
    print(f"METEOR: {meteor_score:.4f}")

    rouge_score, _ = Rouge().compute_score(refs_dict, preds_dict)
    print("ROUGE-L:", round(rouge_score * 100, 2))

    cider_score, _ = Cider().compute_score(refs_dict, preds_dict)
    print("CIDEr:", round(cider_score, 2))

    return (
        round(meteor_score * 100, 2),
        round(rouge_score * 100, 2),
        round(cider_score, 2)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True)
    parser.add_argument("--pred", required=True)
    args = parser.parse_args()

    refs, preds = read_files(args.gold, args.pred)

    compute_bleu(refs, preds)
    compute_other_metrics(refs, preds)