import argparse
import os
from keypoints import OxenHumanKeypointsAnnotation, CocoHumanKeypointsAnnotation
from keypoints import PredictionOutcome
from keypoints import TSVKeypointsDataset
from metrics import Metrics
import matplotlib.pyplot as plt

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
    "-n",
    dest="name",
    required=True,
    type=str,
    help="Name of the chart",
)
parser.add_argument(
    "-t",
    dest="type",
    required=True,
    type=str,
    help="Type of annotations [coco, oxen]",
)
parser.add_argument(
    "-o",
    dest="output",
    required=True,
    type=str,
    help="Directory to store the output stats",
)

args = parser.parse_args()

ground_truth_file = args.ground_truth
predictions_file = args.predictions
output_dir = args.output

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print(f"Loading ground truth... {ground_truth_file}")
ground_truth = TSVKeypointsDataset(annotation_file=ground_truth_file, type=args.type)

print(f"Loading ground truth... {predictions_file}")
predictions = TSVKeypointsDataset(annotation_file=predictions_file, type=args.type)

thresh_vals = []
recall_vals = []
precision_vals = []

for i in range(10):
    thresh = i * 0.1
    thresh_vals.append(thresh)

    joints = OxenHumanKeypointsAnnotation.joints if 'oxen' == args.type else CocoHumanKeypointsAnnotation.joints
    joint_outcomes = {}
    for joint in joints:
        joint_outcomes[joint] = {}
        joint_outcomes[joint][PredictionOutcome.TRUE_POSITIVE] = 0.0
        joint_outcomes[joint][PredictionOutcome.FALSE_POSITIVE] = 0.0
        joint_outcomes[joint][PredictionOutcome.FALSE_NEGATIVE] = 0.0
        joint_outcomes[joint][PredictionOutcome.TRUE_NEGATIVE] = 0.0


    for (input_i, file) in enumerate(ground_truth.list_inputs()):
        gt = ground_truth.get_annotations(file).annotations[0]
        pred = predictions.get_annotations(file).annotations[0]

        outcomes = gt.compute_outcomes(pred, confidence_thresh=thresh)

        for (outcome_i, outcome) in enumerate(outcomes):
            joint = gt.joints[outcome_i]
            joint_outcomes[joint][outcome] += 1.0

    output_file = os.path.join(output_dir, f"results_{thresh}.txt")
    print(f"Writing output to {output_file}")
    
    with open(output_file, "w") as f:
        sum_precision = 0.0
        sum_recall = 0.0
        for joint in joints:
            outcomes = joint_outcomes[joint]
            precision = Metrics.precision(outcomes)
            recall = Metrics.recall(outcomes)

            sum_precision += precision
            sum_recall += recall

            line = (
                f"{joint}\n  Precision@{thresh}: {precision}\n  Recall@{thresh}: {recall}\n"
            )
            print(line)
            f.write(line)

        avg_precision = sum_precision / float(len(joints))
        avg_recall = sum_recall / float(len(joints))
        line = f"\n\nAverage Precision@{thresh}: {avg_precision}\nAverage Recall@{thresh}: {avg_recall}\n"
        print(line)
        f.write(line)
        
        precision_vals.append(avg_precision)
        recall_vals.append(avg_recall)


# plotting the points 
plt.plot(thresh_vals, precision_vals, label="Precision")
plt.plot(thresh_vals, recall_vals, label="Recall")
  
# naming the x axis
plt.xlabel('Threshold')
# naming the y axis
plt.ylabel('Values')
plt.ylim([0, 1.0])

plt.title(f'Precision vs Recall\n{args.name}')

precision_plot_file = os.path.join(output_dir, f"precision.png")
print(f"Saving {precision_plot_file}")
plt.savefig(precision_plot_file)

