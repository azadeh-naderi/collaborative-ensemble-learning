# Proposed Experiments

---

## Experiment 1: Bounded-Degree Communication Graph with Random Neighbor Pairing

**Status:** `[ ] Pending`

### Description

The communication network is a random graph with bounded degree d = 3. At each round, each node is paired with one of its neighbors, chosen uniformly at random. The oracle is a single designated node.

The experiment tests how well the collaborative learning protocol scales and performs under a sparse, degree-constrained topology compared to denser or fully-connected baselines.

### Parameters

| Parameter | Values |
|-----------|--------|
| N (number of agents) | 10, 20 |
| Graph degree bound (d) | 3 |
| Oracle size | 1 node |
| Pairing strategy | Random neighbor (uniform) |

### Goals

- Measure accuracy gain as a function of N under sparse communication.
- Compare convergence behavior between N=10 and N=20.
- Assess the effect of oracle locality (oracle is just another node in the graph, not globally accessible).

### Notes

- The random graph should be generated such that the degree of every node is at most 3 (e.g., a random 3-regular graph or a bounded-degree random graph).
- Pairing is re-randomized each round.
- Run multiple seeds to account for variability in both graph structure and pairing randomness.
