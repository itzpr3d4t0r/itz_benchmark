from data_utils import plot_tests, run_tests, print_info

print_info()

# ===========| CONFIG |===========
DO_TEST = True


def test_setup(curr_size: int, g: dict):
    pass


tests_config = {
    'tests': [],
    'tests_setup': test_setup,
    'max_size': 1000,
    'reps': 100,
    'num': 1,
}

kwargs_dict = {}

plot_config = {
    'title': 'MISSING TITLE',
    'tests': [],
    'mode': 'MIN',
    'limit_to_range': tests_config['max_size'],
    'scatter': True,
    'errbars': False,
    'compare_list': [],
    'x_label': 'MISSING X LABEL',
}

# ===========| END OF CONFIG |===========

if DO_TEST:
    run_tests(**tests_config, **kwargs_dict)

plot_tests(**plot_config)
