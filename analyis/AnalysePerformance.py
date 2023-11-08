import subprocess
import numpy as np
import scipy.stats as sp_stat
import re
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib
matplotlib.rcParams.update({'font.size': 22})


def run_gmh_experiment(
    number_of_fused_simulations, number_of_accepted_proposals, number_of_samples
):
    result = subprocess.run(
        [
            "./gmh",
            f"{number_of_samples}",
            "-n",
            f"{number_of_fused_simulations}",
            "-k",
            f"{number_of_accepted_proposals}",
        ],
        capture_output=True,
    )
    assert result.returncode == 0
    return result.stdout.decode("ASCII")


def analyse_log(log_string):
    log_lines = log_string.split("\n")

    time_re = re.compile("Completed in ([\d.]+) seconds.")
    try:
        execution_time = float(time_re.findall(log_lines[-8])[0])
    except Exception as e:
        print("time failed with line")
        print(log_lines[-8])

    forward_re = re.compile(
        "Executed solver (\d+) times, to evaluate (\d+) forward models."
    )
    try:
        forward = int(forward_re.findall(log_lines[-6])[0][1])
    except Exception as e:
        print("forward failed with line")
        print(log_lines[-6])

    ess_re = re.compile("ESS = +([\d.]+) +([\d.]+)")
    try:
        ess = [float(ess_re.findall(log_lines[-4])[0][i]) for i in range(2)]
    except Exception as e:
        print("ess failed with line")
        print(log_lines[-4])

    acceptance_re = re.compile("Acceptance Ratio \d+/\d+ = (\d+.\d+)")
    try:
        acceptance = float(acceptance_re.findall(log_lines[-3])[0])
    except Exception as e:
        print("acceptance failed with line")
        print(log_lines[-3])

    return execution_time, ess, forward, acceptance


# https://stackoverflow.com/a/10633553
def flatten(lst):
    result = []
    for element in lst:
        if hasattr(element, "__iter__"):
            result.extend(flatten(element))
        else:
            result.append(element)
    return result


def run_experiment_until_significant(n, k, s, confidence=0.95):
    run_experiment = lambda n, k, s: flatten(analyse_log(run_gmh_experiment(n, k, s)))
    met_requirement = False
    experiments = [run_experiment(n, k, s), run_experiment(n, k, s)]
    num_measurements = 2

    while not met_requirement:
        experiments.append(run_experiment(n, k, s))
        num_measurements += 1
        experiments_np = np.array(experiments)
        mean = np.mean(experiments_np[:, :3], axis=0)
        standard_error = sp_stat.sem(experiments_np[:, :3], axis=0)
        interval_from, interval_to = sp_stat.t.interval(
            confidence=confidence,
            df=num_measurements - 1,
            loc=mean,
            scale=standard_error,
        )
        check_from = np.all(interval_from > mean * (1 - 0.5 * confidence))
        check_to = np.all(interval_to < mean * (1 + 0.5 * confidence))
        met_requirement = check_from and check_to
    mean = np.mean(experiments_np, axis=0)
    print(f"{n}, {k}, {s}: found {mean} after {num_measurements} experiments")
    return mean

max_fused = 20
N, K = np.meshgrid(np.arange(1, max_fused + 1), np.arange(1, max_fused + 1))
filename = f"matrix_{max_fused}.npy"

try:
    with open(filename, 'rb') as f:
        matrix = np.load(f)
    print(matrix.shape)
    assert matrix.shape == (max_fused, max_fused, 7)
except:
    matrix = np.empty((max_fused, max_fused, 7))
    matrix[:] = np.nan
    for n in range(1, max_fused + 1):
        for k in range(1, n + 1):
            ess = run_experiment_until_significant(n, k, 1000)
            matrix[n - 1, k - 1, :5] = ess
    matrix[:, :, 5] = matrix[:, :, 1] / matrix[:, :, 0]
    matrix[:, :, 6] = matrix[:, :, 2] / matrix[:, :, 0]
    with open(filename, 'wb') as f:
        np.save(f, matrix)
print(matrix)

#fig = plt.figure(layout="constrained", figsize=(20, 5))
fig = plt.figure(figsize=(20, 5))
axes = fig.subplots(1, 5)
for ax in axes:
    ax.set_aspect("equal", "box")

# Execution time
v_min = np.nanmin(matrix[:, :, 0])
v_max = np.nanmax(matrix[:, :, 0])
cmap_time = matplotlib.colormaps["inferno"]
normalizer = Normalize(v_min, v_max)
axes[0].pcolormesh(N, K, matrix[:, :, 0], cmap=cmap_time, norm=normalizer)
fig.colorbar(
    matplotlib.cm.ScalarMappable(norm=normalizer, cmap=cmap_time),
    ax=axes[0],
    location="bottom",
    label="s",
)

# Number of forward models
v_min = np.nanmin(matrix[:, :, 3])
v_max = np.nanmax(matrix[:, :, 3])
cmap_forward = matplotlib.colormaps["inferno"]
normalizer = Normalize(v_min, v_max)
axes[1].pcolormesh(N, K, matrix[:, :, 3], cmap=cmap_forward, norm=normalizer)
fig.colorbar(
    matplotlib.cm.ScalarMappable(norm=normalizer, cmap=cmap_forward),
    ax=axes[1],
    location="bottom",
    label="# evaluations",
)

# Acceptance ratio
v_min = np.nanmin(matrix[:, :, 4])
v_max = np.nanmax(matrix[:, :, 4])
cmap_acceptance = matplotlib.colormaps["inferno"]
normalizer = Normalize(v_min, v_max)
axes[2].pcolormesh(N, K, matrix[:, :, 4], cmap=cmap_acceptance, norm=normalizer)
fig.colorbar(
    matplotlib.cm.ScalarMappable(norm=normalizer, cmap=cmap_acceptance),
    ax=axes[2],
    location="bottom",
    label="acceptance ratio",
)

# ESS
v_min = np.nanmin(matrix[:, :, 1])
v_max = np.nanmax(matrix[:, :, 1])
cmap_ess = matplotlib.colormaps["inferno"]
normalizer = Normalize(v_min, v_max)
axes[3].pcolormesh(N, K, matrix[:, :, 1], cmap=cmap_ess, norm=normalizer)
fig.colorbar(
    matplotlib.cm.ScalarMappable(norm=normalizer, cmap=cmap_ess),
    #shrink=0.5,
    ax=axes[3],
    location="bottom",
    label="# independent samples",
)

# ESS per time
v_min = np.nanmin(matrix[:, :, 5])
v_max = np.nanmax(matrix[:, :, 5])
cmap_ess_per_time = matplotlib.colormaps["inferno"]
normalizer = Normalize(v_min, v_max)
axes[4].pcolormesh(N, K, matrix[:, :, 5], cmap=cmap_ess_per_time, norm=normalizer)
fig.colorbar(
    matplotlib.cm.ScalarMappable(norm=normalizer, cmap=cmap_ess_per_time),
    #shrink=0.5,
    ax=axes[4],
    location="bottom",
    label="# samples / s",
)

axes_for_headers = fig.subplots(
    1, 5, width_ratios=[1, 1, 1, 1, 1]
)  # , frameon = False)
for ax in axes_for_headers:
    ax.axis("off")
axes_for_headers[0].set_title("Execution time")
axes_for_headers[1].set_title("Forward models")
axes_for_headers[2].set_title("Acceptance Ratio")
axes_for_headers[3].set_title("ESS")
axes_for_headers[4].set_title("ESS per time")

plt.tight_layout()
plt.savefig("gmh-performance.pdf")
plt.show()
