import numpy as np
from scipy import stats


def diebold_mariano(e1: np.ndarray, e2: np.ndarray, h: int = 1, two_sided: bool = True) -> tuple:
    """Diebold-Mariano test for equal predictive accuracy.

    Returns (DM_statistic, p_value). H0: equal accuracy.
    Positive DM means model 1 has larger errors (model 2 is better).
    """
    d = e1 - e2
    n = len(d)
    if n < 2:
        return 0.0, 1.0

    d_bar = np.mean(d)
    variance = np.var(d, ddof=1) / n
    if variance <= 0:
        return 0.0, 1.0

    dm_stat = d_bar / np.sqrt(variance)
    if two_sided:
        p_value = 2.0 * (1.0 - stats.norm.cdf(abs(dm_stat)))
    else:
        p_value = 1.0 - stats.norm.cdf(dm_stat)
    return float(dm_stat), float(p_value)


def paired_ttest(e1: np.ndarray, e2: np.ndarray) -> tuple:
    """Paired t-test for equal mean error. Returns (t_stat, p_value)."""
    t_stat, p_value = stats.ttest_rel(e1, e2)
    return float(t_stat), float(p_value)


def wilcoxon_signed_rank(e1: np.ndarray, e2: np.ndarray) -> tuple:
    """Wilcoxon signed-rank test. Returns (statistic, p_value)."""
    stat, p_value = stats.wilcoxon(e1, e2, alternative='two-sided')
    return float(stat), float(p_value)


def compute_ranking(results: dict, higher_is_better: list = None) -> dict:
    """Rank models by each metric. Lower error is better unless specified."""
    if higher_is_better is None:
        higher_is_better = ["coverage_90"]

    model_names = list(results.keys())
    metrics = list(results[model_names[0]].keys())

    rankings = {m: {} for m in model_names}
    for metric in metrics:
        values = [(m, results[m][metric]) for m in model_names]
        sorted_vals = sorted(values, key=lambda x: x[1], reverse=(metric in higher_is_better))
        for rank, (model, _) in enumerate(sorted_vals, 1):
            rankings[model][metric] = rank

    for m in model_names:
        rankings[m]["avg_rank"] = float(np.mean(list(rankings[m].values())))

    return rankings
