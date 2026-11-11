import numpy as np

from sudoku_rescue.csp import flatten, find_conflicts
from sudoku_rescue.data import Reading

PUZZLE = "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
SOLUTION = "534678912672195348198342567859761423426853791713924856961537284287419635345286179"


def digits(s):
    return [int(c) for c in s]


def is_valid_solution(grid, puzzle=None):
    flat = flatten(grid)
    if 0 in flat or find_conflicts(flat):
        return False
    return puzzle is None or all(p == 0 or p == s for p, s in zip(flatten(puzzle), flat))


def synthetic_reading(true_grid, wrong=None, p_sure=0.99):
    """Fake network output. Every given is read correctly with probability p_sure, except the cells in
    `wrong` = {cell: (misread_digit, p_misread)}: there the wrong digit is the argmax and the TRUE digit
    gets 80% of the remaining probability mass (i.e. it is the second choice)."""
    wrong = wrong or {}
    probs = np.zeros((81, 9))
    grid = list(true_grid)
    for i, d in enumerate(true_grid):
        if not d:
            continue
        if i in wrong:
            rd, pw = wrong[i]
            rest = 1 - pw
            probs[i] = 0.2 * rest / 7
            probs[i, rd - 1] = pw
            probs[i, d - 1] = 0.8 * rest
            grid[i] = rd
        else:
            probs[i] = (1 - p_sure) / 8
            probs[i, d - 1] = p_sure
    return Reading(grid, probs, np.array([d == 0 for d in true_grid]))
