"""Draw the training run from solution3.py.

Nothing here is part of the exercise — the maths all lives in solution3.py, and
this file only reads its output.  It is kept separate so that solution3.py stays
short enough to hold in your head.

Run it with:  python3 solution3_viz.py

Six panels, each answering something problem3.md asks that the printed output
cannot show:

1. Accuracy is finished at step 1; the loss works for 199 more.  Part H's second
   checkpoint and question 3, in one picture.  The loss starts at exactly log 2.
2. Logit, probability, prediction — the three levels problem3.md opens by saying
   people mix up, with this run's own logits laid underneath the sigmoid.
3. Why cross-entropy and not squared error.  Part E asks you to compare the two
   gradients at label 1, logit -20; this plots both across the whole range, and
   the gap there is a factor of half a billion.
4. Where the naive loss dies.  Part F's real lesson: `1 - sigmoid(z)` reaches
   exactly 0.0 at z = 38, which training walks past without noticing, while
   `sigmoid(z)` survives to z = -745.  That asymmetry is why the loss is
   computed from logits and never from probabilities.
5. The boundary and the probability field — the stretch goal.
6. The boundary stopped at step 1; only the confidence sharpened.  The other
   half of the stretch goal, and the answer to "the loss will never reach zero,
   explain what the model is still improving".
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from viz_style import (
    AQUA, AXIS, BLUE, BLUE_RAMP_5, GRID, INK, MUTED, ORANGE, SURFACE,
    draw_line, finish, legend, note, style_axes,
)

# One colour, one meaning.  Blue is the loss, and in the two panels that put
# rival formulas side by side it is always the one we keep — cross-entropy in
# panel 3, the from-logits form in panel 4.  Orange is always the alternative we
# reject.  Aqua is accuracy.
LOSS_COLOR = BLUE
ACCURACY_COLOR = AQUA
KEEP_COLOR = BLUE
REJECT_COLOR = ORANGE

# The two classes.  Marker shape carries the class as well as colour, so the
# scatter still reads with the colour taken away.
CLASS_0_COLOR, CLASS_0_MARKER = ORANGE, "s"
CLASS_1_COLOR, CLASS_1_MARKER = BLUE, "o"

# Probability has a meaningful midpoint — 0.5 is the decision — so the field
# gets a diverging map that is palest exactly at the boundary.
PROBABILITY_FIELD = LinearSegmentedColormap.from_list(
    "probability", [ORANGE, "#f3f0e8", BLUE]
)

ONE_BIT = float(np.log(2))  # 0.6931 nats: the step-0 loss, and Part D's landmark

# Panels 1, 4 and 6 have no empty corner big enough for their note, so those
# notes sit on a soft paper backing rather than directly on the curves.
NOTE_BOX = dict(facecolor=SURFACE, edgecolor="none", alpha=0.82, pad=2.5)


@dataclass(frozen=True)
class Run:
    """One training run, with each quantity as a list ready to plot."""

    steps: list[int]
    losses: list[float]
    accuracies: list[float]
    weights: list[np.ndarray]
    biases: list[float]

    def at(self, step: int) -> SimpleNamespace:
        """The parameters as of `step`, shaped like solution3's BinaryParameters."""
        return SimpleNamespace(weights=self.weights[step], bias=self.biases[step])


def plot_loss_and_accuracy(axes, run: Run) -> None:
    """Panel 1 — the two curves Part H asks you to record, on one pair of axes.

    They part company almost immediately, and that parting is question 3.
    """
    loss_line = draw_line(axes, run.steps, run.losses, LOSS_COLOR,
                          label="cross-entropy (nats)", marker="")[0]
    axes.set_yscale("log")
    axes.set_ylim(0.004, 1.6)
    one_bit = axes.axhline(ONE_BIT, color=MUTED, linestyle="--", linewidth=1,
                           label=f"log 2 = {ONE_BIT:.4f} nats = one bit")
    style_axes(axes, "1 · Accuracy finishes at step 1; the loss runs 199 more",
               "training step", "cross-entropy, nats (log axis)")

    right = axes.twinx()
    accuracy_line = right.plot(
        run.steps, [value * 100 for value in run.accuracies],
        color=ACCURACY_COLOR, linewidth=2, label="accuracy (%)",
    )[0]
    right.set_ylabel("accuracy (%)", color=MUTED, fontsize=9)
    right.set_ylim(40, 116)
    right.tick_params(colors=MUTED, labelsize=8, length=0)
    right.grid(False)
    for spine in right.spines.values():
        spine.set_visible(False)

    perfect = next(step for step, value in zip(run.steps, run.accuracies) if value >= 1.0)
    axes.annotate(
        f"step 0 = {run.losses[0]:.4f} nats = log 2\nlogit 0, sigmoid 0.5, one bit: a coin flip",
        xy=(run.steps[0], run.losses[0]), xytext=(14, -46),
        textcoords="offset points", color=MUTED, fontsize=8.5, bbox=NOTE_BOX,
    )
    right.annotate(
        f"100% at step {perfect}", xy=(perfect, 100), xytext=(86, 7),
        textcoords="offset points", color=ACCURACY_COLOR, fontsize=8.5,
    )
    axes.annotate(
        f"ends at {run.losses[-1]:.4f} nats",
        xy=(run.steps[-1], run.losses[-1]), xytext=(-6, 14),
        textcoords="offset points", color=LOSS_COLOR, fontsize=8.5, ha="right",
    )
    note(axes,
         f"accuracy is one bit per example: once every\n"
         f"point is on the right side of the line it has\n"
         f"nothing left to report.  the loss is a probability,\n"
         f"so it keeps score — falling another {run.losses[perfect] / run.losses[-1]:.0f}x\n"
         f"while accuracy sits at 100%.",
         0.97, 0.54, ha="right", va="top", bbox=NOTE_BOX)
    legend(axes, "lower left", handles=[loss_line, accuracy_line, one_bit])


def plot_three_levels(axes, sigmoid: Callable, logits: np.ndarray,
                      labels: np.ndarray) -> None:
    """Panel 2 — the distinction problem3.md opens with, over this run's logits."""
    grid = np.linspace(-12, 12, 400)
    axes.axvspan(-12, 0, color=CLASS_0_COLOR, alpha=0.07)
    axes.axvspan(0, 12, color=CLASS_1_COLOR, alpha=0.07)
    curve = axes.plot(grid, sigmoid(grid), color=INK, linewidth=2,
                      label="sigmoid: the probability")[0]
    axes.axhline(0.5, color=MUTED, linestyle="--", linewidth=1)
    axes.axvline(0.0, color=AXIS, linewidth=1)

    # The trained logits themselves, as a rug: the abstract curve above, the 200
    # actual examples below it.
    rug_y = -0.085
    handles = []
    for value, color, marker, name in ((0, CLASS_0_COLOR, CLASS_0_MARKER, "label 0"),
                                       (1, CLASS_1_COLOR, CLASS_1_MARKER, "label 1")):
        selected = logits[labels == value]
        handles.append(axes.plot(
            selected, np.full(selected.shape, rug_y), linestyle="none",
            marker=marker, markersize=4, markerfacecolor=color,
            markeredgecolor=SURFACE, markeredgewidth=0.5,
            label=f"{name} — trained logits",
        )[0])

    closest = float(np.abs(logits).min())
    axes.annotate(
        f"closest example: logit {closest:.2f},\nprobability {float(sigmoid(np.array([closest]))[0]):.2f}",
        xy=(closest, rug_y), xytext=(30, 40), textcoords="offset points",
        color=MUTED, fontsize=8.5,
        arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.8),
    )
    axes.set_ylim(-0.16, 1.10)
    style_axes(axes, "2 · Logit, probability, prediction — three levels",
               "logit  =  X @ weights + bias", "probability  =  sigmoid(logit)")
    note(axes, "predict 0\nprobability < 0.5", 0.03, 0.70, color=CLASS_0_COLOR)
    note(axes, "predict 1\nprobability >= 0.5", 0.97, 0.40, color=CLASS_1_COLOR, ha="right")
    note(axes,
         "the loss and the gradient both live\n"
         "on this x-axis, never on the y.  the\n"
         "decision is only the sign of x, which\n"
         "is why one sigmoid neuron is still a\n"
         "linear classifier.",
         0.03, 0.97, va="top")
    legend(axes, "center left", handles=[curve, *handles])


def plot_why_not_squared_error(axes, sigmoid: Callable, wrong_logit: float = -20.0) -> None:
    """Panel 3 — Part E: one confidently wrong example, scored two ways."""
    grid = np.linspace(-20, 20, 400)
    probabilities = sigmoid(grid)
    label = 1.0
    cross_entropy = np.abs(probabilities - label)
    squared_error = np.abs((probabilities - label) * probabilities * (1 - probabilities))

    kept = axes.plot(grid, cross_entropy, color=KEEP_COLOR, linewidth=2,
                     label="cross-entropy:  |p - y|")[0]
    rejected = axes.plot(grid, squared_error, color=REJECT_COLOR, linewidth=2,
                         label="squared error:  |(p - y) · p · (1 - p)|")[0]
    axes.set_yscale("log")
    axes.set_ylim(1e-10, 300)  # headroom above 1, so the legend has a clear band

    at_wrong = float(sigmoid(np.array([wrong_logit]))[0])
    ce_here = abs(at_wrong - label)
    se_here = abs((at_wrong - label) * at_wrong * (1 - at_wrong))
    marker = axes.axvline(wrong_logit, color=MUTED, linestyle="--", linewidth=1,
                          label=f"label 1 at logit {wrong_logit:g} — maximally wrong")
    for value, color, offset in ((ce_here, KEEP_COLOR, (10, -13)),
                                 (se_here, REJECT_COLOR, (8, 5))):
        axes.plot([wrong_logit], [value], linestyle="none", marker="o", markersize=6,
                  markerfacecolor=color, markeredgecolor=SURFACE, markeredgewidth=1.5)
        axes.annotate(f"{value:.2g}", xy=(wrong_logit, value), xytext=offset,
                      textcoords="offset points", color=color, fontsize=8.5)
    style_axes(axes, "3 · Why cross-entropy and not squared error",
               "logit, for an example whose label is 1", "|dloss/dlogit| (log axis)")
    note(axes,
         f"at logit {wrong_logit:g} the two gradients\n"
         f"differ by a factor of {ce_here / se_here:.0e}.\n"
         "squared error keeps p·(1-p), the\n"
         "sigmoid's own derivative, and it dies\n"
         "at both ends — so the worse it is\n"
         "wrong, the less it learns, which is\n"
         "backwards.  cross-entropy's log is\n"
         "that factor's exact reciprocal and\n"
         "cancels it, leaving p - y.",
         0.22, 0.40, va="top", bbox=NOTE_BOX)
    legend(axes, "upper right", handles=[kept, rejected, marker])


def plot_where_the_naive_loss_dies(axes, sigmoid: Callable,
                                   binary_cross_entropy_from_logits: Callable) -> None:
    """Panel 4 — Part F.  A stable sigmoid does not save you; leaving logit space kills you."""
    grid = np.arange(0.0, 60.0, 0.25)
    honest = sigmoid(-grid)      # what 1 - sigmoid(z) really is
    naive = 1.0 - sigmoid(grid)  # what float64 hands you if you compute it that way

    true_line = axes.plot(grid, honest, color=KEEP_COLOR, linewidth=2,
                          label="what 1 - sigmoid(z) really is")[0]
    naive_line = axes.plot(grid, naive, color=REJECT_COLOR, linewidth=2,
                           label="1.0 - sigmoid(z) in float64")[0]
    axes.set_yscale("log")
    axes.set_ylim(1e-24, 3)

    # Part F asks for these two integers by name, so find them rather than say them.
    positive = np.arange(1.0, 200.0)
    cliff = float(positive[(1.0 - sigmoid(positive)) <= 0][0])
    negative = -np.arange(1.0, 800.0)
    survives_to = float(negative[sigmoid(negative) > 0][-1])

    edge = axes.axvline(cliff, color=MUTED, linestyle="--", linewidth=1,
                        label=f"z = {cliff:g}: it becomes exactly 0.0")
    axes.axhline(np.finfo(np.float64).eps, color=GRID, linewidth=1)
    axes.annotate(
        "float64 machine epsilon, 2.2e-16 — the\nlast gap it can represent below 1.0",
        xy=(2.0, np.finfo(np.float64).eps), xytext=(4, 28),
        textcoords="offset points", color=MUTED, fontsize=8,
    )
    axes.annotate(
        "log(0.0) = -inf.\nthe loss dies here.",
        xy=(cliff, 1e-16), xytext=(14, 72), textcoords="offset points",
        color=REJECT_COLOR, fontsize=8.5,
    )
    style_axes(axes, "4 · Where the naive loss dies",
               "logit z", "1 - sigmoid(z)  (log axis)")

    at_800 = float(binary_cross_entropy_from_logits(np.array([800.0]), np.array([0.0])))
    note(axes,
         f"sigmoid(z) itself survives all the way\n"
         f"to z = {survives_to:g}, because float64 reaches\n"
         f"down to 1e-324 — but it can only\n"
         f"resolve 2.2e-16 *below one*.  so 'just\n"
         f"under 1' runs out about {survives_to / -cliff:.0f}x sooner, at\n"
         f"z = {cliff:g}, and training walks past {cliff:g} inside\n"
         f"a few hundred steps.  from logits there\n"
         f"is no cliff: logaddexp(0, 800) = {at_800:g} nats.",
         0.03, 0.04, bbox=NOTE_BOX)
    legend(axes, "upper right", handles=[true_line, naive_line, edge])


def plot_boundary_and_field(axes, run: Run, forward: Callable, x: np.ndarray,
                            labels: np.ndarray, early_step: int = 1) -> None:
    """Panel 5 — the stretch goal: the points, the boundary, and the field."""
    pad = 1.0
    grid_x = np.linspace(x[:, 0].min() - pad, x[:, 0].max() + pad, 240)
    grid_y = np.linspace(x[:, 1].min() - pad, x[:, 1].max() + pad, 240)
    mesh_x, mesh_y = np.meshgrid(grid_x, grid_y)
    points = np.column_stack([mesh_x.ravel(), mesh_y.ravel()])
    _, field = forward(points, run.at(-1))
    field = field.reshape(mesh_x.shape)

    axes.contourf(mesh_x, mesh_y, field, levels=np.linspace(0, 1, 41),
                  cmap=PROBABILITY_FIELD, alpha=0.55)
    rings = axes.contour(mesh_x, mesh_y, field, levels=[0.01, 0.1, 0.9, 0.99],
                         colors=[MUTED], linewidths=0.8, linestyles=":")
    axes.clabel(rings, inline=True, fontsize=7, fmt="p = %g", colors=MUTED)

    # The step-1 boundary is drawn thick and pale *underneath* the final one, so
    # the pale halo around the dark line is the evidence that it never moved.
    span = np.array([grid_x[0], grid_x[-1]])
    boundaries = []
    for step, color, width in ((early_step, BLUE_RAMP_5[0], 6), (-1, BLUE_RAMP_5[4], 2)):
        weights, bias = run.weights[step], run.biases[step]
        name = f"boundary at step {run.steps[step]}"
        boundaries.append(axes.plot(span, -(weights[0] * span + bias) / weights[1],
                                    color=color, linewidth=width, label=name)[0])

    classes = []
    for value, color, marker, name in ((0, CLASS_0_COLOR, CLASS_0_MARKER, "class 0"),
                                       (1, CLASS_1_COLOR, CLASS_1_MARKER, "class 1")):
        selected = x[labels == value]
        classes.append(axes.plot(selected[:, 0], selected[:, 1], linestyle="none",
                                 marker=marker, markersize=5, markerfacecolor=color,
                                 markeredgecolor=SURFACE, markeredgewidth=1,
                                 label=name)[0])

    final_weights = run.weights[-1]
    axes.annotate(
        "", xy=tuple(final_weights), xytext=(0, 0),
        arrowprops=dict(arrowstyle="-|>", color=INK, linewidth=1.8, shrinkA=0, shrinkB=0),
    )
    axes.annotate(
        f"weights = ({final_weights[0]:.2f}, {final_weights[1]:.2f}): points from class 0\n"
        f"toward class 1, at right angles to the boundary.\n"
        f"bias = {run.biases[-1]:.3f}, near zero, because the two\n"
        f"clusters sit either side of the origin.",
        xy=tuple(final_weights), xytext=(10, -46), textcoords="offset points",
        color=INK, fontsize=8.5,
    )
    axes.set_xlim(grid_x[0], grid_x[-1])
    axes.set_ylim(grid_y[0], grid_y[-1])
    axes.set_aspect("equal")  # otherwise the panel's shape fakes the angle
    style_axes(axes, "5 · The boundary, and the probability field around it",
               "feature 1", "feature 2")
    legend(axes, "upper left", frameon=True, facecolor=SURFACE, edgecolor="none",
           framealpha=0.85, handles=[*boundaries, *classes])


def plot_confidence_sharpens(axes, run: Run, sigmoid: Callable, x: np.ndarray,
                             steps=(1, 5, 20, 50)) -> None:
    """Panel 6 — the same boundary seen edge-on, as it steepens."""
    final_weights = run.weights[-1]
    direction = final_weights / np.linalg.norm(final_weights)
    distances = np.linspace(-6, 6, 400)

    chosen = list(steps) + [len(run.steps) - 1]
    for step, color in zip(chosen, BLUE_RAMP_5):
        weights, bias = run.weights[step], run.biases[step]
        logits = distances * float(weights @ direction) + bias
        axes.plot(distances, sigmoid(logits), color=color, linewidth=2,
                  label=f"step {run.steps[step]} — |w| = {np.linalg.norm(weights):.2f}")

    axes.axhline(0.5, color=MUTED, linestyle="--", linewidth=1)
    axes.axvline(0.0, color=AXIS, linewidth=1)

    # The example sitting closest to the boundary, and all it gains from 199 steps.
    final_logits = x @ final_weights + run.biases[-1]
    margin = float(np.abs(final_logits).min() / np.linalg.norm(final_weights))
    first, last = run.at(chosen[0]), run.at(chosen[-1])
    began = float(sigmoid(np.array([margin * float(first.weights @ direction) + first.bias]))[0])
    ended = float(sigmoid(np.array([margin * float(last.weights @ direction) + last.bias]))[0])
    axes.plot([margin, margin], [began, ended], color=INK, linewidth=1)
    for value in (began, ended):
        axes.plot([margin], [value], linestyle="none", marker="o", markersize=5,
                  markerfacecolor=INK, markeredgecolor=SURFACE, markeredgewidth=1.2)
    axes.annotate(
        f"the closest example is {margin:.2f} away.\n"
        f"the model never changes its\n"
        f"mind — {began:.2f} sure becomes {ended:.2f} sure.",
        xy=(margin, (began + ended) / 2), xytext=(0.62, 0.44),
        textcoords="axes fraction", color=INK, fontsize=8.5,
        arrowprops=dict(arrowstyle="-", color=INK, linewidth=0.8),
    )
    axes.set_ylim(-0.04, 1.10)
    style_axes(axes, "6 · The boundary stops; confidence sharpens",
               "signed distance from the boundary", "probability")
    note(axes,
         "every curve crosses 0.5 in the\n"
         "same place — after one step the\n"
         "boundary was already within half\n"
         "a degree of its final angle, and\n"
         "it never moved again.  all 199\n"
         "later steps went into growing |w|,\n"
         "the steepness of this ramp and\n"
         "nothing else.  that is what the\n"
         "falling loss buys, and why it\n"
         "never reaches zero: certainty\n"
         "would need |w| = infinity.",
         0.03, 0.97, va="top", bbox=NOTE_BOX)
    legend(axes, "lower right")


def show_training_report(
    x: np.ndarray,
    labels: np.ndarray,
    sigmoid: Callable,
    binary_cross_entropy_from_logits: Callable,
    forward: Callable,
    loss_and_gradients: Callable,
    learning_rate: float = 0.2,
    steps: int = 200,
) -> Path:
    """Draw six views of one training run.  Hand it the functions you wrote."""

    # solution3's history records the step, loss and accuracy but not the
    # parameters, and panels 5 and 6 are entirely about where the parameters
    # went.  So replay the same loop here using your own loss_and_gradients —
    # no maths is redone by hand, and solution3.py stays exactly as it is.
    weights, bias = np.zeros(x.shape[1]), 0.0
    run = Run([], [], [], [], [])
    for step in range(steps + 1):
        parameters = SimpleNamespace(weights=weights, bias=bias)
        loss, gradient_weights, gradient_bias = loss_and_gradients(x, labels, parameters)
        _, probabilities = forward(x, parameters)
        run.steps.append(step)
        run.losses.append(float(loss))
        run.accuracies.append(float(np.mean((probabilities >= 0.5).astype(np.int64) == labels)))
        run.weights.append(weights.copy())
        run.biases.append(float(bias))
        weights = weights - learning_rate * gradient_weights
        bias = bias - learning_rate * gradient_bias

    final_logits, _ = forward(x, run.at(-1))

    figure, axes = plt.subplots(2, 3, figsize=(16.5, 9), facecolor=SURFACE)
    figure.suptitle(
        f"One neuron deciding which of two clusters {len(x)} points came from",
        color=INK, fontsize=14, fontweight="bold", x=0.012, ha="left",
    )
    plot_loss_and_accuracy(axes[0][0], run)
    plot_three_levels(axes[0][1], sigmoid, final_logits, labels)
    plot_why_not_squared_error(axes[0][2], sigmoid)
    plot_where_the_naive_loss_dies(axes[1][0], sigmoid, binary_cross_entropy_from_logits)
    plot_boundary_and_field(axes[1][1], run, forward, x, labels)
    plot_confidence_sharpens(axes[1][2], run, sigmoid, x)
    figure.tight_layout(rect=(0, 0, 1, 0.955))

    return finish(figure, Path(__file__).with_name("solution3_training.png"))


if __name__ == "__main__":
    from solution3 import (
        binary_cross_entropy_from_logits, forward, loss_and_gradients,
        make_binary_blobs, sigmoid,
    )

    x, labels = make_binary_blobs()
    figure_path = show_training_report(
        x, labels, sigmoid, binary_cross_entropy_from_logits, forward, loss_and_gradients,
    )
    print(f"\nSaved the training graphs to {figure_path}")
