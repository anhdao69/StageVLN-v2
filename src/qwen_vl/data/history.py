"""Causal history selection; counts always include the current observation."""
def history_indices(t, mode, recent=4):
    if t < 0 or recent < 0:
        raise ValueError('Negative timestep/history')
    if mode == 'uniform8':
        return list(range(t+1)) if t <= 8 else [i*t//8 for i in range(9)]
    if mode == 'recent':
        return list(range(max(0,t-recent),t+1))
    if mode == 'current':
        return [t]
    raise ValueError(mode)
