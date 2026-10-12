"""
Matrix Chain Multiplication Optimizer (interval dynamic programming)

Input  : n, then n + 1 positive integers d0 ... dn (matrix Ai is d(i-1) x di), n <= 500
Output : minimum number of scalar multiplications, optimal parenthesization,
         and a comparison with naive left-to-right multiplication.

Run    : python3 matrix_chain.py < input.txt
"""
import sys
import time

MAX_MATRICES = 500


def read_input():
    """Module 1: read n and the n + 1 dimensions from standard input and validate them."""
    data = sys.stdin.read().split()
    if not data:
        raise ValueError("No input given.")
    n = int(data[0])
    dims = list(map(int, data[1:2 + n]))
    if n < 1:
        raise ValueError("Need at least one matrix.")
    if n > MAX_MATRICES:
        raise ValueError(f"At most {MAX_MATRICES} matrices are allowed.")
    if len(dims) != n + 1:
        raise ValueError(f"Expected {n + 1} dimensions, got {len(dims)}.")
    if any(d <= 0 for d in dims):
        raise ValueError("All dimensions must be positive integers.")
    return n, dims


def optimize(dims):
    """Module 2: fill the DP tables. m[i][j] = minimum cost of Ai..Aj, s[i][j] = best split k."""
    n = len(dims) - 1
    m = [[0] * (n + 1) for _ in range(n + 1)]
    s = [[0] * (n + 1) for _ in range(n + 1)]
    for length in range(2, n + 1):              # chain length
        for i in range(1, n - length + 2):      # start of the chain
            j = i + length - 1                  # end of the chain
            best = float("inf")
            for k in range(i, j):               # try every split point
                cost = m[i][k] + m[k + 1][j] + dims[i - 1] * dims[k] * dims[j]
                if cost < best:
                    best = cost
                    s[i][j] = k
            m[i][j] = best
    return m, s


def parenthesize(s, i, j):
    """Module 3: rebuild the optimal bracketing of Ai..Aj from the split table s."""
    if i == j:
        return f"A{i}"
    k = s[i][j]
    return "(" + parenthesize(s, i, k) + " x " + parenthesize(s, k + 1, j) + ")"


def naive_cost(dims):
    """Module 4a: cost of ((A1 x A2) x A3) x ... x An, multiplying strictly left to right."""
    return sum(dims[0] * dims[k] * dims[k + 1] for k in range(1, len(dims) - 1))


def naive_order(n):
    """Module 4b: the bracketing used by the naive strategy."""
    order = "A1"
    for i in range(2, n + 1):
        order = f"({order} x A{i})"
    return order


def main():
    n, dims = read_input()
    start = time.time()
    m, s = optimize(dims)
    elapsed = time.time() - start
    best, naive = m[1][n], naive_cost(dims)
    print(f"Matrices      : {n}")
    print(f"Optimal cost  : {best}")
    if n <= 20:                                  # long chains are too wide to print
        print(f"Optimal order : {parenthesize(s, 1, n)}")
        print(f"Naive order   : {naive_order(n)}")
    print(f"Naive cost    : {naive}")
    saved = naive - best
    print(f"Savings       : {saved} ({saved / naive * 100:.2f}% fewer multiplications)")
    print(f"Speed-up      : {naive / best:.2f}x")
    print(f"DP time       : {elapsed:.3f} s")


if __name__ == "__main__":
    try:
        main()
    except ValueError as err:
        print(f"Error: {err}")
        sys.exit(1)
