import json
from statistics import mean, stdev, median, geometric_mean
from timeit import repeat
from typing import List, Tuple, Literal, Callable, Optional

from matplotlib import pyplot as plt
from tqdm import tqdm

__all__ = ["Plotter", "Evaluator", "plot_tests", "run_tests"]

JSON_DIR = "files_json"


class Plotter:
    filters_map = {
        "MEAN": mean,
        "MIN": min,
        "MAX": max,
        "MEDIAN": median,
        "GEOMEAN": geometric_mean,
    }

    def __init__(
        self,
        title: str,
        tests: List[Tuple[str, str]],
        mode: Literal["MIN", "MAX", "MEAN", "MEDIAN", "GEOMEAN"] = "MIN",
        limit_to_range: int = -1,
        x_label: str = "NULL",
    ):
        plt.style.use("dark_background")
        self.title = title
        self.tests = tests
        self.mode = mode
        self.mode_func = self.filters_map[mode]
        self.limit_to_range = limit_to_range
        self.x_label = x_label

    def plot_tests(self, scatter: bool = False, errbars: bool = False) -> None:
        for file_name, color in self.tests:
            data = self._load_data(file_name)
            if data:
                timings = self._extract_timings(data)
                self._print_statistics(file_name, data, timings)
                self._plot_timings(timings, file_name, color, scatter, errbars)
        self._finalize_plot()

    def _load_data(self, file_name: str) -> Optional[dict]:
        """Load data from a JSON file."""
        try:
            with open(f"{JSON_DIR}/{file_name}.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"File {file_name}.json not found!")
            return None

    def _extract_timings(self, data: dict) -> List[float]:
        """Extract and filter the timings from the data."""
        return [self.mode_func(dp) for dp in data["data"][: self.limit_to_range]]

    def _print_statistics(
        self, file_name: str, data: dict, timings: List[float]
    ) -> None:
        """Print basic statistics about the timings."""
        print(f"=== {file_name} ===")
        print(f"Total: {sum(sum(dp) for dp in data['data'])}")
        print(f"Mean: {mean(timings):.6f}")
        print(f"Median: {median(timings):.6f}")
        print(f"Stdev: {stdev(timings):.6f}\n")

    def _plot_timings(
        self, timings: List[float], label: str, color: str, scatter: bool, errbars: bool
    ) -> None:
        """Plot the timings on the graph."""
        if scatter:
            plt.scatter(range(len(timings)), timings, color=color, label=label, s=0.5)
        else:
            plt.plot(timings, color=color, label=label, linewidth=1)

        if errbars:
            plt.errorbar(
                range(len(timings)),
                timings,
                yerr=stdev(timings),
                fmt="none",
                ecolor=color,
                alpha=0.2,
            )

    def _finalize_plot(self) -> None:
        """Finalize and show the plot."""
        plt.legend()
        plt.title(self.title)
        plt.xlabel(self.x_label)
        plt.ylabel("Time (s)")
        plt.show()

    def compare(
        self, indices: List[Tuple[int, int]], c1: str = "white", c2: str = "lime"
    ) -> None:
        num_plots = len(indices)
        for idx, (i1, i2) in enumerate(indices):
            data_1 = self._load_data(self.tests[i1][0])
            data_2 = self._load_data(self.tests[i2][0])
            if data_1 and data_2:
                timings_1 = self._extract_timings(data_1)
                timings_2 = self._extract_timings(data_2)
                self._plot_comparison(
                    timings_1, timings_2, i1, i2, idx, num_plots, c1, c2
                )
        plt.show()

    def _plot_comparison(
        self,
        timings_1: List[float],
        timings_2: List[float],
        i1: int,
        i2: int,
        idx: int,
        num_plots: int,
        c1: str,
        c2: str,
    ) -> None:
        """Plot a comparison between two sets of timings."""
        plt.subplot(num_plots, 2, idx * 2 + 1)
        plt.scatter(
            range(len(timings_1)), timings_1, color=c1, label=self.tests[i1][0], s=0.5
        )
        plt.scatter(
            range(len(timings_2)), timings_2, color=c2, label=self.tests[i2][0], s=0.5
        )
        plt.legend()
        if idx == 0:
            plt.title("Timings")
        if idx == num_plots - 1:
            plt.ylabel("Time (s)")
            plt.xlabel(self.x_label)

        plt.subplot(num_plots, 2, idx * 2 + 2)
        comparative_data = [
            100 * ((t1 / t2) - 1) for t1, t2 in zip(timings_1, timings_2)
        ]
        plt.scatter(range(len(comparative_data)), comparative_data, color="red", s=1)
        plt.axhline(0, color="violet", linewidth=2)
        if idx == 0:
            plt.title("Relative % improvement")
        if idx == num_plots - 1:
            plt.xlabel(self.x_label)


class Evaluator:
    def __init__(
        self,
        tests: List[Tuple[str, str]],
        tests_setup: Callable[[int, dict], None],
        max_size: int = 1000,
        reps: int = 30,
        num: int = 1,
    ):
        self.tests = tests
        self.tests_setup = tests_setup
        self.max_size = max_size
        self.reps = reps
        self.num = num
        self.G = {}

    def run(self) -> None:
        for test_name, statement in self.tests:
            self._run_single_test(test_name, statement)

    def _run_single_test(self, test_name: str, statement: str) -> None:
        """Run a single test and store the results."""
        data = {
            "title": test_name,
            "settings": self._get_settings(statement),
            "data": [],
        }
        print(f"\n========| {test_name.upper()} |========")
        progress_bar = tqdm(total=self.max_size, ncols=100, colour="green")

        for curr_size in range(1, self.max_size + 1):
            self._run_iteration(curr_size, statement, data)
            progress_bar.update(1)

        progress_bar.close()
        self._save_results(test_name, data)

    def _get_settings(self, statement: str) -> dict:
        """Return the settings for the current test."""
        return {
            "statement": statement,
            "max_size": self.max_size,
            "reps": self.reps,
            "num": self.num,
        }

    def _run_iteration(self, curr_size: int, statement: str, data: dict) -> None:
        """Run a single iteration of the test."""
        self.G["curr_size"] = curr_size
        self.tests_setup(curr_size, self.G)
        data["data"].append(self.run_test(statement))

    @staticmethod
    def _save_results(test_name: str, data: dict) -> None:
        """Save the results of the test to a JSON file."""
        with open(f"{JSON_DIR}/{test_name}.json", "w") as f:
            json.dump(data, f)

    def inject(self, **kwargs) -> None:
        self.G.update(kwargs)

    def run_test(self, statement: str) -> List[float]:
        return repeat(statement, globals=self.G, number=self.num, repeat=self.reps)

    def __enter__(self) -> "Evaluator":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        pass


def plot_tests(
    title: str,
    tests: List[Tuple[str, str]],
    mode: Literal["MIN", "MAX", "MEAN", "MEDIAN", "GEOMEAN"] = "MIN",
    limit_to_range: int = -1,
    scatter: bool = False,
    errbars: bool = False,
    compare_list: Optional[List[Tuple[int, int]]] = None,
    x_label: str = "NULL",
) -> None:
    if not tests:
        print(
            "Nothing to plot... (Have you filled the 'tests' field correctly in your plot configuration?)"
        )
        return

    plotter = Plotter(title, tests, mode, limit_to_range, x_label)
    plotter.plot_tests(scatter, errbars)
    if compare_list:
        plotter.compare(compare_list)


def run_tests(
    tests: List[Tuple[str, str]],
    tests_setup: Callable[[int, dict], None],
    max_size: int = 1000,
    reps: int = 30,
    num: int = 1,
    **kwargs,
) -> None:
    if not tests:
        print(
            "Nothing to benchmark... (Have you filled the 'tests' field correctly in your configuration?)"
        )
        return

    with Evaluator(tests, tests_setup, max_size, reps, num) as evaluator:
        if kwargs:
            evaluator.inject(**kwargs)
        evaluator.run()


def print_info():
    print("Hello and welcome to ItzBenchmark!\n")


def package_configuration(
    tests_config: dict, environment_kwargs: dict, globals: dict
) -> dict:
    environment_kwargs.update(globals)
    return {"config": tests_config, "e_kwargs": environment_kwargs}


def run_suite(run_bench: bool, packaged_data: dict, run_plot: bool, plot_config: dict):
    print_info()

    if run_bench:
        run_tests(**packaged_data["config"], **packaged_data["e_kwargs"])
    if run_plot:
        print("\n=====| Available Data Filters |=====")
        for filter_name in Plotter.filters_map.keys():
            print(f"{filter_name} |", end=" ")
        print(f"(Selected: {plot_config['mode']})")
        plot_tests(**plot_config)
