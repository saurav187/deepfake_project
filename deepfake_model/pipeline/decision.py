"""The same cumulative evidence is available at file end and every live update."""
import numpy as np


def trajectory(scores, aggregation):
    scores = np.asarray(scores, dtype=np.float64)
    if aggregation == 'fraction':
        scores = scores >= .5
    elif aggregation != 'mean':
        raise ValueError(aggregation)
    return np.cumsum(scores) / np.arange(1, len(scores)+1)


def summarize(scores, ends, rule):
    values = trajectory(scores, rule['aggregation'])
    if not len(values):
        return dict(score=None, verdict='INSUFFICIENT', alerts=0, first_alert_s=None)
    flagged = values >= rule['threshold']
    starts = np.flatnonzero(flagged & ~np.r_[False, flagged[:-1]])
    return dict(score=float(values[-1]), verdict='FAKE' if flagged[-1] else 'REAL',
                alerts=len(starts), first_alert_s=float(ends[starts[0]]) if len(starts) else None)
