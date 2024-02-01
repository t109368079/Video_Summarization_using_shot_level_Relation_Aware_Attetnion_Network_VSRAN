from ortools.algorithms import pywrapknapsack_solver
import math

solver = pywrapknapsack_solver.KnapsackSolver(
    pywrapknapsack_solver.KnapsackSolver.
    KNAPSACK_DYNAMIC_PROGRAMMING_SOLVER, 'knapsack')
# KNAPSACK_DYNAMIC_PROGRAMMING_SOLVER
def solve_knapsack(values, weights, capacities):
    values = [value+0.001 for value in values]      # 避免最小值為0
    if math.ceil(1/min(values)) < 1000:
      scale= 1000
    else:
      scale = math.ceil(1/min(values))
    # print(scale)
    # print(values,end='')
    values = [value * scale for value in values]
    # print('capacities',capacities)
    solver.Init(values, weights, capacities)
    total_value = solver.Solve()
    packed_item_indexes = [index for index in range(len(values)) if solver.BestSolutionContains(index)]
    if len(packed_item_indexes) == 1:
      print(packed_item_indexes,values)
    return packed_item_indexes

if __name__ == '__main__':
    pass
