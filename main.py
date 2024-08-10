from data_utils import run_suite

# ===========| CONFIG |===========
RUN_BENCHMARK = True
PLOT = True

# Import your current config here:
from configs.empty_configuration import configuration_data

plot_config = {
    'title': 'MISSING TITLE',      # Title for the plot
    'tests': [],                   # List of test results to plot
    'mode': 'MIN',                 # Mode for plotting (e.g., MIN for minimum values)
    'limit_to_range': -1,          # Limit for the x-axis range
    'scatter': True,               # Whether to use a scatter plot
    'errbars': False,              # Whether to show error bars on the plot
    'compare_list': [],            # List of test cases to compare
    'x_label': 'MISSING X LABEL',  # Label for the x-axis
}

# ===========| END OF CONFIG |===========

run_suite(RUN_BENCHMARK, configuration_data, PLOT, plot_config)