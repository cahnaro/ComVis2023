# Ground truth data
ground_truth = [1, 1, 1, 0, 0, 1, 0, 1, 1, 0]

# Algorithm detection results
detection_results = [1, 1, 0, 0, 1, 1, 0, 1, 0, 0]

tp = 0
fp = 0
tn = 0
fn = 0

for i in range(len(ground_truth)):
    if ground_truth[i] == 1 and detection_results[i] == 1:
        tp += 1
    elif ground_truth[i] == 0 and detection_results[i] == 1:
        fp += 1
    elif ground_truth[i] == 0 and detection_results[i] == 0:
        tn += 1
    elif ground_truth[i] == 1 and detection_results[i] == 0:
        fn += 1

precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1_score = 2 * precision * recall / (precision + recall)

print('Precision:', precision)
print('Recall:', recall)
print('F1-score:', f1_score)
