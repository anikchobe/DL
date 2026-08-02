import timeit
import numpy as np


# Large enough to make the timing difference visible.
N = 1_000_000

# Build both inputs outside the timed functions.
# Otherwise, we would partly be measuring array/list construction.
python_data = [float(i) for i in range(N)]
numpy_data = np.arange(N, dtype=np.float64)


def regular_python():
    """
    For every x:
        y = 1.5*x + 2
        add y^2 to the total
    """
    total = 0.0

    for x in python_data:
        y = 1.5 * x + 2.0
        total += y * y

    return total


def numpy_version():
    """
    Perform the same computation using vectorized NumPy operations.
    """
    y = 1.5 * numpy_data + 2.0
    return np.sum(y * y)


# Verify that both versions produce effectively the same answer.
python_result = regular_python()
numpy_result = numpy_version()

print(f"Regular Python result: {python_result}")
print(f"NumPy result:        {numpy_result}")
print(f"Results match:       {np.isclose(python_result, numpy_result)}")
print()


# Run each function several times.
# timeit.repeat returns one total time for each repeat.
number_of_runs = 3
number_of_repeats = 5

python_times = timeit.repeat(
    regular_python,
    number=number_of_runs,
    repeat=number_of_repeats,
)

numpy_times = timeit.repeat(
    numpy_version,
    number=number_of_runs,
    repeat=number_of_repeats,
)

# Use the best observed timing to reduce interference from background activity.
python_time = min(python_times) / number_of_runs
numpy_time = min(numpy_times) / number_of_runs

print(f"Regular Python: {python_time:.6f} seconds per run")
print(f"NumPy:          {numpy_time:.6f} seconds per run")
print(f"Speedup:        {python_time / numpy_time:.1f}x")
