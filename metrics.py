import math
from collections import Counter
import evaluate


def compute_bleu(candidate, reference, max_order=2):
    cand_tokens = candidate.split()
    ref_tokens = reference.split()

    precisions = []
    for n in range(1, max_order + 1):
        cand_ngrams = Counter([tuple(cand_tokens[i:i+n]) for i in range(len(cand_tokens) - n + 1)])
        ref_ngrams = Counter([tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens) - n + 1)])

        overlap = {ng: min(count, ref_ngrams[ng]) for ng, count in cand_ngrams.items()}
        p_n = sum(overlap.values()) / max(1, sum(cand_ngrams.values()))
        precisions.append(p_n)

    if any(p == 0 for p in precisions):
        return 0.0

    # Brevity Penalty
    c, r = len(cand_tokens), len(ref_tokens)
    BP = 1 if c >= r else math.exp(1 - r / c)

    log_avg = sum(math.log(p) for p in precisions) / max_order
    bleu = BP * math.exp(log_avg)
    return bleu

def lcs(X, Y):
    m, n = len(X), len(Y)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m):
        for j in range(n):
            if X[i] == Y[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
    return dp[m][n]

def rouge_l(candidate, reference):
    cand_tokens, ref_tokens = candidate.split(), reference.split()
    lcs_len = lcs(cand_tokens, ref_tokens)

    precision = lcs_len / len(cand_tokens)
    recall = lcs_len / len(ref_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def main(bleu=False, rouge=False):
    candidate_bleu = ''
    reference_bleu = ''
    n = 3
    candidate_rouge = ''
    reference_rouge = ''
    if bleu:
        print("Your BLEU: ", compute_bleu(candidate_bleu, reference_bleu, max_order=n))
        reference_bleu = evaluate.load("bleu", max_order=n)
        results = reference_bleu.compute(predictions=[candidate_bleu], references=[reference_bleu], tokenizer=lambda x: x.split(), max_order=n)
        print("BLEU (ref): ", results["bleu"])

    if rouge:
        print("Your Rouge-L: ", rouge_l(candidate_rouge, reference_rouge)[-1])
        reference_rouge = evaluate.load('rouge')
        print("Ref Rouge-L: ", reference_rouge.compute(predictions=[candidate_rouge], references=[reference_rouge], tokenizer=lambda x: x.split())['rougeL'])

    return 0


if __name__ == '__main__':
    main(bleu=True, rouge=True)
