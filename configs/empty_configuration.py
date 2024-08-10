from data_utils import package_configuration


def test_setup(curr_size: int, g: dict) -> None:
    """Function to set up the environment at each curr_size increment up to max_size.
       curr_size gets automatically incremented once per run up to max_size. This can
       be useful for when you want to test the same function for say an increasingly larger input size."""
    pass


tests_config = {
    "tests": [],                # List of tests to run and benchmark
    "tests_setup": test_setup,  # Function to set up the environment at each curr_size increment up to max_size
    "max_size": 100,            # Maximum size (x) for the tests
    "reps": 1,                  # Number of repetitions for each test
    "num": 1,                   # number of 'calls' per test
}

kwargs_dict = {}  # Contains pairs of name: value for extra information about the test


# This is what you want to import in main.py to be able to run this configuration
configuration_data = package_configuration(tests_config, kwargs_dict, globals())
