
from oxen.metrics.outcome import PredictionOutcome

class Metrics:
    def precision(outcomes):
        tp = outcomes[PredictionOutcome.TRUE_POSITIVE]
        fp = outcomes[PredictionOutcome.FALSE_POSITIVE]
        precision = 0.0 if (tp == 0.0 and fp == 0.0) else tp / (tp + fp)
        # print(f"Precision = {tp} / ({tp + fp}) = {precision}")
        return precision

    def recall(outcomes):
        tp = outcomes[PredictionOutcome.TRUE_POSITIVE]
        fn = outcomes[PredictionOutcome.FALSE_NEGATIVE]
        recall = 0.0 if (tp == 0.0 and fn == 0.0) else tp / (tp + fn)
        # print(f"Recall = {tp} / ({tp + fn}) = {recall}")
        return recall
