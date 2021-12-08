"""
Includes evaluation metrics for evaluating models
"""
from rouge_score import rouge_scorer

def calculate_rouge(ground_truth, prediction):
    """
    Calulates Rouge1 score based on passed reference and hypothesis (predictions)
    """
    scorer = rouge_scorer.RougeScorer(['rouge1'])
    scores = scorer.score(" ".join(ground_truth), " ".join(prediction))
    return scores

# def calculate_rouge(reference, hypothesis):
#     scorer = rouge_scorer.RougeScorer(['rouge1'])
#     scores = scorer.score(" ".join(reference), " ".join(hypothesis))
#     return scores