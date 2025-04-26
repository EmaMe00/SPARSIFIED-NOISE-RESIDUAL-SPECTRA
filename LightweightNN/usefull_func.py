from package import *

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight, gain=1.0)
            if m.bias is not None:
                init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            init.ones_(m.weight)
            init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, 0, 0.1)
            init.zeros_(m.bias)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Function to calculate classification metrics
def calculate_metrics(y_true, y_pred):
    # Compute the confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Compute accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    # Compute additional metrics
    error_rate = (fp + fn) / (tp + tn + fp + fn)
    precision_pos = precision_score(y_true, y_pred, pos_label=1)
    precision_neg = precision_score(y_true, y_pred, pos_label=0)
    recall_pos = recall_score(y_true, y_pred, pos_label=1)
    recall_neg = recall_score(y_true, y_pred, pos_label=0)
    fpr_val = fp / (fp + tn)  # False Positive Rate
    fnr_val = fn / (fn + tp)  # False Negative Rate
    f1_pos = f1_score(y_true, y_pred, pos_label=1)
    f1_neg = f1_score(y_true, y_pred, pos_label=0)

    return accuracy, tn, fp, fn, tp, error_rate, precision_pos, precision_neg, recall_pos, recall_neg, fpr_val, fnr_val, f1_pos, f1_neg
