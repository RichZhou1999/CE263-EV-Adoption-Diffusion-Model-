
import numpy as np


def generate_age_value():
    return np.random.randint(0, 80)


def generate_income_value():
    return np.random.normal(10000, 5000)


def generate_zipcode_value():
    return 94707


value_generator_dict = {
    "age": generate_age_value,
    "income": generate_income_value,
    "zipcode": generate_zipcode_value,
}
