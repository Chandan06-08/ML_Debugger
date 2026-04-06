def detect_issues(profile, analysis):
    issues = []

    gap = analysis.get("gap", 0)
    train_acc = analysis.get("train_acc", 0)
    val_acc = analysis.get("val_acc", 0)

    # Overfitting
    if gap > 0.1:
        issues.append("overfitting")

    # Underfitting
    if train_acc < 0.6 and val_acc < 0.6:
        issues.append("underfitting")

    # Low accuracy
    if val_acc < 0.7:
        issues.append("low accuracy")

    # Class imbalance
    class_dist = profile.get("class_distribution", {})
    if class_dist:
        values = list(class_dist.values())
        if min(values) > 0 and max(values) / min(values) > 2:
            issues.append("class imbalance")
    else:
        issues.append("class distribution unknown")

    # Default case
    if not issues:
        issues.append("no major issues detected")

    return issues