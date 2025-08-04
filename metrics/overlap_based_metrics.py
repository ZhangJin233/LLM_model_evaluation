"""
demo_metrics.py
计算 BLEU、ROUGE-1/2/4、ROUGE-L 与 METEOR
"""

from typing import List
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer, scoring
from nltk import word_tokenize
from tqdm import tqdm


def _tokenize(text: str) -> List[str]:
    return word_tokenize(text.lower())


def bleu(references: List[str], hypotheses: List[str]) -> float:
    refs_tok = [[_tokenize(r)] for r in references]  # 每个句子可支持多参考
    hyps_tok = [_tokenize(h) for h in hypotheses]
    smooth = SmoothingFunction().method4
    return corpus_bleu(refs_tok, hyps_tok, smoothing_function=smooth)


def rouge(r_list: List[str], h_list: List[str], n: int = 2) -> float:
    scorer = rouge_scorer.RougeScorer([f"rouge{n}"], use_stemmer=True)
    aggregator = scoring.BootstrapAggregator()
    for ref, hyp in zip(r_list, h_list):
        aggregator.add_scores(scorer.score(ref, hyp))
    return aggregator.aggregate()[f"rouge{n}"].mid.fmeasure


def rouge_l(r_list: List[str], h_list: List[str]) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    aggregator = scoring.BootstrapAggregator()
    for ref, hyp in zip(r_list, h_list):
        aggregator.add_scores(scorer.score(ref, hyp))
    return aggregator.aggregate()["rougeL"].mid.fmeasure


def meteor(r_list: List[str], h_list: List[str]) -> float:
    scores = []
    for ref, hyp in zip(r_list, h_list):
        scores.append(
            meteor_score([_tokenize(ref)], _tokenize(hyp))  # 注意: 外层 list 表示多参考
        )
    return sum(scores) / len(scores)


def evaluate(references: List[str], hypotheses: List[str]) -> None:
    assert len(references) == len(hypotheses), "长度不一致"
    print(f"BLEU       : {bleu(references, hypotheses):.4f}")
    for n in (1, 2, 3, 4):
        print(f"ROUGE-{n:<2}  : {rouge(references, hypotheses, n):.4f}")
    print(f"ROUGE-L    : {rouge_l(references, hypotheses):.4f}")
    print(f"METEOR     : {meteor(references, hypotheses):.4f}")


if __name__ == "__main__":
    refs = ["the cat is on the mat", "there is a cat on the mat"]
    hyps = ["the cat sat on the mat", "a cat is on the mat"]
    evaluate(refs, hyps)
