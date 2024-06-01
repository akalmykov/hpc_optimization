import pygad
import numpy

import matplotlib
matplotlib.use('TkAgg')

predicted_workload = [
    50, 20, 10, 60, 110, 20, 10, 30, 90, 40,
    20, 70, 60, 60, 150, 70, 50, 40, 50, 70,
    90, 50, 80, 100, 90, 10, 20, 20, 30, 50,
    40, 50, 80, 60, 20, 70
]


salary = 3000
onboarding = 3000
severance = 3000
contractor_pay = 2*salary
longer_contract_discount = lambda span: span*contractor_pay*min(span*0.05, 0.4)

def fitness_func(ga_instance, solution, solution_idx):
    contractors_required = numpy.maximum(predicted_workload - solution, [0] * len(solution))
    contractors_cost = numpy.sum(contractors_required * contractor_pay)

    contractors_discount = 0
    contractor_span = 0
    for i in range(len(contractors_required)-1):
        if contractors_required[i] > 0 and contractors_required[i+1] > 0:
            contractor_span += min(contractors_required[i], contractors_required[i+1])
        elif contractor_span > 0:
            contractors_discount += longer_contract_discount(contractor_span)
            contractor_span = 0

    payroll = numpy.sum(solution) * salary
    staff_changes = numpy.diff(solution)
    onboarding_costs = numpy.sum(numpy.maximum(staff_changes, [0] * len(staff_changes)) * onboarding)
    severance_costs = numpy.sum(numpy.minimum(staff_changes, [0] * len(staff_changes)) * -severance)

    return 1.0/(contractors_cost - contractors_discount + onboarding_costs + severance_costs + payroll + 0.000001)

num_generations = 30000
num_parents_mating = 15

sol_per_pop = 30
num_genes = len(predicted_workload)

ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       sol_per_pop=sol_per_pop,
                       gene_type=int,
                       mutation_num_genes=1, ## to reduce variation in the outcome
                       num_genes=num_genes,
                       fitness_func=fitness_func,
                       gene_space=numpy.arange(1, max(predicted_workload)+1), #range(1, max(predicted_workload)+10),
                       random_seed=1,
                       # stop_criteria=['saturate_100'],
                       # keep_elitism=3
                       # parent_selection_type='nsga2'
                       )

ga_instance.run()

solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)
print(f"Predicted Workload : {predicted_workload}")
for i in predicted_workload:
    print(i)
print(f"Number of employees : {solution}")
for i in solution:
    print(i)

contractors = numpy.maximum(predicted_workload - solution, [0] * len(solution))
print(f"Number of contractors : {contractors}")
for i in contractors:
    print(i)

print(f"Fitness value of the best solution = {solution_fitness}")

prediction = numpy.sum(numpy.array(predicted_workload)*solution)
print(f"Predicted output 1 based on the best solution : {prediction}")
