import math
from collections import Counter
from typing import List, Sequence


def _tokenize(sent: str) -> List[str]:
    # 这里用最朴素的空格切分，你可以换成自己的分词器
    return sent.lower().split()


def _ngrams(tokens: Sequence[str], n: int):
    return zip(*(tokens[i:] for i in range(n)))


def _modified_precision(refs, hyp, n):
    hyp_ngrams = Counter(_ngrams(hyp, n))
    max_ref_counts = Counter()
    for ref in refs:
        ref_counts = Counter(_ngrams(ref, n))
        for ng, cnt in ref_counts.items():
            max_ref_counts[ng] = max(max_ref_counts[ng], cnt)

    clipped = {ng: min(cnt, max_ref_counts[ng]) for ng, cnt in hyp_ngrams.items()}
    return sum(clipped.values()), max(1, sum(hyp_ngrams.values()))


def bleu(references: List[str], hypotheses: List[str], max_n: int = 4):
    refs_tok = [[_tokenize(r)] for r in references]
    hyps_tok = [_tokenize(h) for h in hypotheses]

    p_num, p_den = [0] * max_n, [0] * max_n
    for ref_list, hyp in zip(refs_tok, hyps_tok):
        for n in range(1, max_n + 1):
            num, den = _modified_precision(ref_list, hyp, n)
            p_num[n - 1] += num
            p_den[n - 1] += den

    # Chen & Cherry method‑4 smoothing (mirrors NLTK implementation)
    precisions = []
    incvnt = 1
    hyp_total_len = sum(len(h) for h in hyps_tok)

    for i in range(max_n):
        if p_num[i] == 0 and hyp_total_len > 1:
            # numerator = 1 / (2^incvnt * k / ln(len(T))); NLTK default k = 5
            numerator = 1 / (2 ** incvnt * 5 / math.log(hyp_total_len))
            precisions.append(numerator / p_den[i])
            incvnt += 1
        else:
            precisions.append(p_num[i] / p_den[i] if p_den[i] else 0)

    log_p = sum(math.log(p) for p in precisions) / max_n
    ref_len = sum(len(r[0]) for r in refs_tok)
    hyp_len = sum(len(h) for h in hyps_tok)
    bp = 1 if hyp_len > ref_len else math.exp(1 - ref_len / hyp_len)

    return bp * math.exp(log_p)


if __name__ == "__main__":
    refs = ["the cat is on the mat", "there is a cat on the mat"]
    hyps = ["the cat on mat", "a cat is on the mat"]

    print(f"Pure-Python BLEU: {bleu(refs, hyps):.4f}")

    # 校对：与 NLTK corpus_bleu 一致
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

    refs_tok = [[r.lower().split()] for r in refs]
    hyps_tok = [h.lower().split() for h in hyps]
    score_nltk = corpus_bleu(
        refs_tok, hyps_tok, smoothing_function=SmoothingFunction().method4
    )
    print(f"NLTK corpus_bleu: {score_nltk:.4f}")
