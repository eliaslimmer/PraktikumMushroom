from scipy.stats import spearmanr #For the spearman correlation computation

def iou(span1, span2):
    start1, end1 = span1
    start2, end2 = span2
    intersection = max(0, min(end1, end2) - max(start1, start2))
    union = max(end1, end2) - min(start1, start2)
    return intersection / union if union > 0 else 0.0

def average_iou(preds, gts):
    ious = []
    for gt_span in gts:
        max_iou = max([iou(gt_span, pred_span) for pred_span in preds], default=0.0)
        ious.append(max_iou)
    return sum(ious) / len(ious) if ious else 0.0

def span_to_dict(span_list):
    return {(s['start'], s['end']): s['prob'] for s in span_list}

def spearman_score(soft_pred, soft_gt):
    pred_dict = {(s['start'], s['end']): s['prob'] for s in soft_pred}
    gt_dict = {(s['start'], s['end']): s['prob'] for s in soft_gt}
    
    common_keys = list(set(pred_dict.keys()) & set(gt_dict.keys()))
    if not common_keys:
        return None

    pred_scores = [pred_dict[k] for k in common_keys]
    gt_scores = [gt_dict[k] for k in common_keys]

    if len(set(gt_scores)) <= 1 or len(set(pred_scores)) <= 1:
        return None #if all the probabilities are same then it doesn't work

    corr, _ = spearmanr(pred_scores, gt_scores)
    return corr

def evaluate_predictions(predictions, hard_references, soft_references):
    all_ious = []
    all_spearman = []

    for sample_pred in predictions:
        sample_id = sample_pred['id']
        pred_spans = sample_pred['hard_labels']
        soft_pred = sample_pred['soft_labels']

        gt_spans = hard_references[sample_id]
        soft_gt = soft_references[sample_id]

        iou_score = average_iou(pred_spans, gt_spans)
        all_ious.append(iou_score)

        spearman = spearman_score(soft_pred, soft_gt)
        if spearman is not None:
            all_spearman.append(spearman)

    return {
        "mean_iou": sum(all_ious) / len(all_ious),
        "mean_spearman": sum(all_spearman) / len(all_spearman)
    }