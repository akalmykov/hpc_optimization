from math import ceil

import numpy as np
from scipy.optimize import minimize, LinearConstraint

CPU_MODEL_COUNT = 5
GPU_MODEL_COUNT = 2
cpu_cores = [32, 40, 48, 52, 56, 0, 0]
gpu_cores = [0, 0, 0, 0, 0, 6912, 16896]

DEBUG_OUTPUT = 0

# Number of GPUs demand in a given year
# Y1: 600; 	Y2: 2000; 	Y3: 1000
#
# Number of CPUs demand in a given year
# Y1: 930; 	Y2: 900; 	Y3: 100
# CPU to GPU ratio is 1:20

core_demand = [930*24, 900*24, 100*24, 600*6912, 2000*6912, 1000*6912]

def pv_costs(x, years, discount_rate):
    return x * ((1+discount_rate) ** years)


def pv_revenue(x, years, discount_rate):
    return x / ((1+discount_rate) ** years)


def f(x):
    risk_free_rate = 0.05
    discount_rate = 0.13
    tax_rate = 0.17
    prices = [7995, 8750, 9900, 11590, 12980, 7668, 43989]
    core_count = [32, 40, 48, 52, 56, 6912, 16896]
    average_cpu_core_price = np.sum(prices[0:5]) / np.sum(core_count[0:5])
    average_gpu_core_price = np.sum(prices[5:7]) / np.sum(core_count[5:7])


    year_1_costs = np.dot(prices, x[0:7])
    year_2_costs = np.dot(np.dot(prices, 0.9), x[7:14])
    year_3_costs = np.dot(np.dot(prices, 0.81), x[14:21])

    cu_costs = pv_costs(year_3_costs, 2, risk_free_rate) + pv_costs(year_2_costs, 1, risk_free_rate) + year_1_costs

    cores_available = x * (core_count * 3)
    # profit per core

    revenue_per_cpu_core = 178.40
    revenue_per_gpu_core = 4.38

    y1_cpu_revenue = calc_revenue(np.dot(x[0:7], cpu_cores), core_demand[0], revenue_per_cpu_core, discount_rate, 1)
    y1_gpu_revenue = calc_revenue(np.dot(x[0:7], gpu_cores), core_demand[3], revenue_per_gpu_core, discount_rate, 1)

    y2_cpu_revenue = calc_revenue(np.dot(x[0:14], cpu_cores * 2), core_demand[1], revenue_per_cpu_core, discount_rate, 2)
    y2_gpu_revenue = calc_revenue(np.dot(x[0:14], gpu_cores * 2), core_demand[4], revenue_per_gpu_core, discount_rate, 2)

    y3_cpu_revenue = calc_revenue(np.dot(x[0:21], cpu_cores * 3), core_demand[2], revenue_per_cpu_core, discount_rate, 3)
    y3_gpu_revenue = calc_revenue(np.dot(x[0:21], cpu_cores * 3), core_demand[5], revenue_per_gpu_core, discount_rate, 3)

    revenue = y1_cpu_revenue + y1_gpu_revenue + y2_cpu_revenue + y2_gpu_revenue + y3_cpu_revenue + y3_gpu_revenue
    # revenue = 0
    avg_cpu_core_maint = 5
    avg_gpu_core_maint = 1

    y1_cpu_sunk = calc_sunk_and_maintenance(np.dot(x[0:7], cpu_cores), core_demand[0], average_cpu_core_price * 0.1, avg_cpu_core_maint, discount_rate, 1)
    y1_gpu_sunk = calc_sunk_and_maintenance(np.dot(x[0:7], gpu_cores), core_demand[3], average_gpu_core_price * 0.1, avg_gpu_core_maint, discount_rate, 1)

    y2_cpu_sunk = calc_sunk_and_maintenance(np.dot(x[0:14], cpu_cores * 2), core_demand[1], average_cpu_core_price * 0.15, avg_cpu_core_maint, discount_rate, 2)
    y2_gpu_sunk = calc_sunk_and_maintenance(np.dot(x[0:14], gpu_cores * 2), core_demand[4], average_gpu_core_price * 0.15, avg_gpu_core_maint, discount_rate, 2)

    y3_cpu_sunk = calc_sunk_and_maintenance(np.dot(x[0:21], cpu_cores * 3), core_demand[2], average_cpu_core_price * 0.2, avg_cpu_core_maint, discount_rate, 3)
    y3_gpu_sunk = calc_sunk_and_maintenance(np.dot(x[0:21], cpu_cores * 3), core_demand[5], average_gpu_core_price * 0.2, avg_gpu_core_maint, discount_rate, 3)

    sunk_and_maintenance_costs = y1_cpu_sunk + y1_gpu_sunk + y2_cpu_sunk + y2_gpu_sunk + y3_cpu_sunk + y3_gpu_sunk

    # tax shield
    useful_life = 5
    y1_ddb = 0
    y2_ddb = 0
    y3_ddb = 0
    for i in range(1,3):
        if year_1_costs - y1_ddb > 0:
            y1_ddb += (2/useful_life) * (year_1_costs - y1_ddb)
            y1_ddb += (2/useful_life) * (year_2_costs - y2_ddb)
            y1_ddb += (2/useful_life) * (year_3_costs - y3_ddb)


    if DEBUG_OUTPUT == 1:
        print(f"costs: {cu_costs + sunk_and_maintenance_costs}")
        print(f"revenue: {(revenue - (revenue-y1_ddb-y2_ddb-y3_ddb)*tax_rate)}")
    return cu_costs - (revenue - (revenue-y1_ddb-y2_ddb-y3_ddb)*tax_rate) + sunk_and_maintenance_costs


def calc_revenue(cores_available_, core_demand_, revenue_per_core, discount_rate, year):
    cores_used_ = min(cores_available_, core_demand_)
    return pv_revenue(cores_used_ * revenue_per_core, year, discount_rate)


def calc_sunk_and_maintenance(cores_available_, core_demand_, fixed_maintenance_per_core, sunk_deprecation_fixed, discount_rate, year):
    unused_cores = max(cores_available_ - core_demand_, 0)
    return pv_costs(unused_cores * (fixed_maintenance_per_core + sunk_deprecation_fixed), year, discount_rate)


def get_product_name(index):
    names = ["Intel CPU Max 9462", "Intel CPU Max 9460", "Intel CPU Max 9468", "Intel CPU Max 9470", "Intel CPU Max 9480", "NVIDIA A100", "NVIDIA H100"]
    return names[index % len(names)]

if __name__ == '__main__':
    # Starting point
    x_start = np.array([5] * 3 * 7)
    empty_year = [0, 0, 0, 0, 0, 0, 0]

    positive_constraint = []
    for i in range(21):
        ith_var = [0] * 21
        ith_var[i] = 1
        positive_constraint.append(ith_var)

    # non_linear_eq = lambda x:
    #
    # non_linear_constr = NonlinearConstraint(
    #     non_linear_eq, 0.2, np.inf)

    constraint_expression = [  # , 0, 0, 0, 0, 0, 0, 0
        cpu_cores + empty_year + empty_year,  # year 1 CPU cores
        cpu_cores + cpu_cores + empty_year,  # year 2 CPU cores total
        cpu_cores + cpu_cores + cpu_cores,  # year 3 CPU cores total
        gpu_cores + empty_year + empty_year,  # year 1 GPU cores
        gpu_cores + gpu_cores + empty_year,  # year 2 GPU cores  total
        gpu_cores + gpu_cores + gpu_cores,  # year 3 GPU cores total

    ]
    constraint_expression.extend(positive_constraint)

    # for i in range(len(core_demand)):
    #     core_demand[i] = core_demand[i]*0.8
    greater_than_zero = [0, 0, 0, 0, 0, 0]

    linear_constraint = LinearConstraint(
        constraint_expression,
        core_demand + [0.1] * 21,  # minimum cores requirement + positive constraint
        # greater_than_zero + [0.1] * 21,  # positive constraint
        [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf] + [np.inf] * 21  # maximum cores requirement + positive constraint
    )
    bounds = [(0, None)] * 3 * 7
    result = minimize(f, x_start, method='trust-constr', constraints=linear_constraint)

    result.x.round(1)
    print(result.x)
    x_ceiled = []
    for i, v in enumerate(result.x):
        if i % 7 == 0:
            print(f"Year {i // 7}")
        c_v = ceil(v) if v > 1 else 0
        print(get_product_name(i) + ": " + str(c_v))
        x_ceiled.append(c_v)

    all_cores = [32, 40, 48, 52, 56, 6912, 16896]
    print(f"Year 1, CPU core demand {core_demand[0]}, cores available: {ceil(np.dot(x_ceiled[0:7], cpu_cores))}")
    print(f"Year 1, GPU core demand {core_demand[3]}, cores available: {ceil(np.dot(x_ceiled[0:7], gpu_cores))}")
    print(f"Year 2, CPU core demand {core_demand[1]}, cores available: {ceil(np.dot(x_ceiled[0:14], cpu_cores * 2))}")
    print(f"Year 2, GPU core demand {core_demand[4]}, cores available: {ceil(np.dot(x_ceiled[0:14], gpu_cores * 2))}")
    print(f"Year 3, CPU core demand {core_demand[2]}, cores available: {ceil(np.dot(x_ceiled[0:21], cpu_cores * 3))}")
    print(f"Year 3, GPU core demand {core_demand[5]}, cores available: {ceil(np.dot(x_ceiled[0:21], gpu_cores * 3))}")
