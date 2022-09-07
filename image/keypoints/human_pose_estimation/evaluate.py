import argparse
from keypoints import OxenHumanKeypointsAnnotation
from keypoints import PredictionOutcome
from keypoints import TSVKeypointsDataset
from metrics import Metrics

parser = argparse.ArgumentParser()
parser.add_argument(
    "-g",
    dest="ground_truth",
    required=True,
    type=str,
    help="The ground truth annotations file",
)
parser.add_argument(
    "-p",
    dest="predictions",
    required=True,
    type=str,
    help="The predictions from the model",
)
parser.add_argument(
    "-o",
    dest="output",
    required=True,
    type=str,
    help="Where to store the output stats",
)
parser.add_argument(
    "-t",
    dest="threshold",
    default=0.5,
    type=float,
    help="Confidence threshold",
)

args = parser.parse_args()

ground_truth_file = args.ground_truth
predictions_file = args.predictions
thresh = args.threshold

print(f"Loading ground truth... {ground_truth_file}")
ground_truth = TSVKeypointsDataset(annotation_file=ground_truth_file)

print(f"Loading ground truth... {predictions_file}")
predictions = TSVKeypointsDataset(annotation_file=predictions_file)

joints = OxenHumanKeypointsAnnotation.joints
joint_outcomes = {}
for joint in joints:
    joint_outcomes[joint] = {}
    joint_outcomes[joint][PredictionOutcome.TRUE_POSITIVE] = 0.0
    joint_outcomes[joint][PredictionOutcome.FALSE_POSITIVE] = 0.0
    joint_outcomes[joint][PredictionOutcome.FALSE_NEGATIVE] = 0.0
    joint_outcomes[joint][PredictionOutcome.TRUE_NEGATIVE] = 0.0


for (input_i, file) in enumerate(ground_truth.list_inputs()):
    gt = ground_truth.get_annotation(file).annotations[0]
    pred = predictions.get_annotation(file).annotations[0]

    outcomes = gt.compute_outcomes(pred, confidence_thresh=thresh)
    
    for (outcome_i, outcome) in enumerate(outcomes):
        joint = gt.joints[outcome_i]
        joint_outcomes[joint][outcome] += 1.0

print(f"Writing output to {args.output}")
with open(args.output, "w") as f:
    sum_precision = 0.0
    sum_recall = 0.0
    for joint in joints:
        outcomes = joint_outcomes[joint]
        precision = Metrics.precision(outcomes)
        recall = Metrics.recall(outcomes)

        sum_precision += precision
        sum_recall += recall

        line = f"{joint}\n  Precision@{thresh}: {precision}\n  Recall@{thresh}: {recall}\n"
        print(line)
        f.write(line)
        
    avg_precision = sum_precision / float(len(joints))
    avg_recall = sum_recall / float(len(joints))
    line = f"\n\nAverage Precision@{thresh}: {avg_precision}\nAverage Recall@{thresh}: {avg_recall}\n"
    print(line)
    f.write(line)