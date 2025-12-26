import numpy as np
from mcts_ipc import PyArena

arena = PyArena("example_mcts", num_slots=1024, num_handlers=1)

while True:
    sv = arena.pop_ready_view(handler=0, clear_outputs=True)

    # Input information
    action_mask = np.asarray(sv.action_mask())  # shape (b, NUM_ACTIONS)
    done = np.sum(np.asarray(sv.obj0())) == 0  # shape (b, MAX_OBJ0)

    # Output information
    priors = np.asarray(sv.priors())            # shape (b, NUM_ACTIONS)
    values = np.asarray(sv.values())            # shape (b,)            

    priors[:] = 0.0

    b, n = priors.shape
    for i in range(b):
        norm = np.sum(action_mask[i])
        for j in range(n):
            if action_mask[i, j]:
                priors[i, j] = 1.0 / norm
        if done:
            values[i] = 1.0
        else:
            values[i] = -1.0

    sv.mark_done()  # or arena.mark_done(sv.slot)
