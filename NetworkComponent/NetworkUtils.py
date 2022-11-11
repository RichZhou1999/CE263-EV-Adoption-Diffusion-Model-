import numpy as np
import bisect


def generate_age_value(*args,**kwargs):
    return np.random.randint(0, 80)


def generate_income_value(*args,**kwargs):
    return np.random.normal(10000, 5000)


def generate_zipcode_value(*args,**kwargs):
    return 94707


def generate_income_with_prob_value_list(prob_list=[], value_list=[], *args, **kwargs):
    if len(prob_list) !=len(value_list):
        raise('probality and value not ')
    sum_prob_list = []
    temp_sum = 0
    for prob in prob_list:
        temp_sum += prob
        sum_prob_list.append(temp_sum)
    r = np.random.random()
    index = bisect.bisect(sum_prob_list, r)
    if index < len(sum_prob_list):
        return value_list[index]
    else:
        return value_list[-1]