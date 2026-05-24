"""Draw the training run from solution2.py.

Nothing here is part of the exercise — the maths all lives in solution2.py, and
this file only reads its output.  It is kept separate so that solution2.py stays
short enough to hold in your head.

Run it with:  python3 solution2.py

The graph:
1. The mean loss falls — 11.0 → 2.4e-16.
2. The line moves onto the data — the five points, plus the fitted line at steps 0/1/2/3/20 in one hue light→dark.
This is the panel that makes "the model is a line" concrete.
3. Two knobs, one bowl — stage 1's parabola becomes a contour map over (weight, bias), with the descent path and the optimum starred.
Aspect is locked to equal, which I checked renders exactly (37.64 px per unit on both axes) — otherwise
the panel's shape would fake the bowl's shape, and the bowl's shape is the whole point.
4. Weight and bias find the line — landing on 3.000 and 2.000.
5. Every example votes — per-example bars of error and error × x, with dashed lines at their means, which are the two gradients (−2 and −6).
It shows the middle example contributing nothing to the weight, since its x is 0 — that's problem2.md's question 1 and its stated goal
about shared parameters, in one picture.
6. Your line-37 question, answered. At rate 0.9 the weight wobbles and the bias doesn't because the weight's gradient is scaled by mean(x²) = 2
and the bias's by 1. So each step moves the weight 0.9 × 2 = 1.8 of the way to the bottom — past 1, so it overshoots and lands on the far side
while the bias moves 0.9 and just settles. (Past 2.0 the weight would diverge.) 
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

from viz_style import (
    AQUA, BLUE, BLUE_RAMP_5, INK, MUTED, ORANGE, SURFACE,
    draw_line, finish, legend, note, style_axes,
)

# One colour, one meaning, in every panel: blue is the loss, orange is the
# weight, aqua is the bias.
LOSS_COLOR = BLUE
WEIGHT_COLOR = ORANGE
BIAS_COLOR = AQUA

# Loss is a magnitude, so it gets one hue from light (small) to dark (large).
LOSS_FIELD = LinearSegmentedColormap.from_list("loss", BLUE_RAMP_5)


@dataclass(frozen=True)
class Run:
    """One training run, with each quantity as a list ready to plot."""

    steps: list[int]
    weights: list[float]
    biases: list[float]
    losses: list[float]


def plot_loss(axes, run: Run) -> None:
    """Panel 1 — one number for the whole batch, and it goes down."""
    draw_line(axes, run.steps, run.losses, LOSS_COLOR)
    style_axes(axes, "1 · The mean loss falls", "training step", "mean loss")
    axes.annotate(
        f"starts at {run.losses[0]:.1f}\nweight 0, bias 0 predicts nothing but 0",
        xy=(run.steps[0], run.losses[0]), xytext=(14, -16),
        textcoords="offset points", color=MUTED, fontsize=8.5,
    )
    axes.annotate(
        f"ends at {run.losses[-1]:.1e}",
        xy=(run.steps[-1], run.losses[-1]), xytext=(-6, 18),
        textcoords="offset points", color=MUTED, fontsize=8.5, ha="right",
    )


def plot_fitted_lines(axes, run: Run, x: np.ndarray, targets: np.ndarray,
                      early_steps=(0, 1, 2, 3)) -> None:
    """Panel 2 — what the model actually *is*: a line, moving onto the data."""
    span = np.array([x.min() - 0.4, x.max() + 0.4])
    chosen = [step for step in early_steps if step < run.steps[-1]] + [run.steps[-1]]
    for step, color in zip(chosen, BLUE_RAMP_5):
        axes.plot(span, run.weights[step] * span + run.biases[step],
                  color=color, linewidth=2, label=f"step {step}")
    axes.plot(x, targets, linestyle="none", marker="o", markersize=7,
              markerfacecolor=INK, markeredgecolor=SURFACE, markeredgewidth=2,
              label="the five examples")
    style_axes(axes, "2 · The line moves onto the data", "x", "target")
    note(axes, "the targets have no noise in them, so a\n"
               "perfect fit exists: by the last step the line\n"
               "passes exactly through all five points.",
         0.97, 0.06, ha="right")
    legend(axes, "upper left", ncol=2, columnspacing=1.2)


def plot_loss_surface(axes, run: Run, loss_at: Callable[[float, float], float],
                      best_weight: float, best_bias: float) -> None:
    """Panel 3 — stage 1's bowl, now with two knobs: a contour map."""
    weight_grid = np.linspace(best_weight - 6, best_weight + 6, 100)
    bias_grid = np.linspace(best_bias - 4, best_bias + 4, 100)
    surface = np.array([[loss_at(weight, bias) for weight in weight_grid]
                        for bias in bias_grid])
    rings = axes.contour(weight_grid, bias_grid, surface,
                         levels=[0.5, 2, 4, 8, 16, 32], cmap=LOSS_FIELD, linewidths=1.2)
    axes.clabel(rings, inline=True, fontsize=7, fmt="%g", colors=MUTED)

    path = draw_line(axes, run.weights, run.biases, WEIGHT_COLOR,
                     label="every (weight, bias) it tried")[0]
    best = axes.plot([best_weight], [best_bias], linestyle="none", marker="*",
                     markersize=13, markerfacecolor=INK, markeredgecolor=SURFACE,
                     markeredgewidth=1.5, label="the bottom of the bowl")[0]
    axes.annotate(
        "starts at (0, 0)", xy=(run.weights[0], run.biases[0]), xytext=(-4, -18),
        textcoords="offset points", color=WEIGHT_COLOR, fontsize=8.5, ha="center",
    )
    axes.set_aspect("equal")  # otherwise the panel shape fakes the bowl's shape
    style_axes(axes, "3 · Two knobs, one bowl", "weight", "bias")
    legend(axes, "upper left", frameon=True, facecolor=SURFACE, edgecolor="none",
           framealpha=0.9,
           handles=[Line2D([], [], color=BLUE, linewidth=1.2, label="rings of equal loss"),
                    path, best])


def plot_parameters(axes, run: Run, best_weight: float, best_bias: float) -> None:
    """Panel 4 — the two numbers you would report, and where they land."""
    draw_line(axes, run.steps, run.weights, WEIGHT_COLOR, label="weight")
    draw_line(axes, run.steps, run.biases, BIAS_COLOR, label="bias")
    axes.axhline(best_weight, color=MUTED, linestyle="--", linewidth=1,
                 label=f"where they belong ({best_weight:g} and {best_bias:g})")
    axes.axhline(best_bias, color=MUTED, linestyle="--", linewidth=1)
    for values, color, name in ((run.weights, WEIGHT_COLOR, "weight"),
                                (run.biases, BIAS_COLOR, "bias")):
        axes.annotate(
            f"{name} {values[-1]:.3f}", xy=(run.steps[-1], values[-1]), xytext=(-6, -18),
            textcoords="offset points", color=color, fontsize=8.5, ha="right",
        )
    axes.set_ylim(-0.4, 5.6)  # headroom for the legend
    style_axes(axes, "4 · Weight and bias find the line", "training step", "value")
    legend(axes, "upper right")


def plot_batch_votes(axes, x: np.ndarray, errors: np.ndarray,
                     gradient_weight: float, gradient_bias: float) -> None:
    """Panel 5 — the whole point of a batch: five examples, two shared knobs."""
    positions = np.arange(len(x))
    bias_votes = axes.bar(positions - 0.19, errors, width=0.34, color=BIAS_COLOR,
                          label="error  (its vote on the bias)")
    weight_votes = axes.bar(positions + 0.19, errors * x, width=0.34, color=WEIGHT_COLOR,
                            label="error × x  (its vote on the weight)")
    bias_mean = axes.axhline(gradient_bias, color=BIAS_COLOR, linestyle="--", linewidth=1,
                             label=f"their mean = dloss/dbias = {gradient_bias:g}")
    weight_mean = axes.axhline(gradient_weight, color=WEIGHT_COLOR, linestyle="--", linewidth=1,
                               label=f"their mean = dloss/dweight = {gradient_weight:g}")
    axes.set_xticks(positions, [f"x = {value:g}" for value in x])
    axes.set_ylim(top=max(errors.max(), 0) + 11)  # headroom for the legend
    style_axes(axes, "5 · Every example votes, at the first step", "example", "contribution")
    note(axes, "each knob is shared by all five examples, so each\n"
               "gradient is the average of what the five ask for.\n"
               "the middle example has no say in the weight at all —\n"
               "its x is 0, so its vote is multiplied away.",
         0.03, 0.06)
    legend(axes, "upper right", handles=[bias_votes, weight_votes, bias_mean, weight_mean])


def plot_two_curvatures(axes, run: Run, curvature: float, learning_rate: float,
                        best_weight: float, best_bias: float) -> None:
    """Panel 6 — the question written in solution2.py, answered."""
    draw_line(axes, run.steps, run.weights, WEIGHT_COLOR, label="weight", marker="")
    draw_line(axes, run.steps, run.biases, BIAS_COLOR, label="bias", marker="")
    for value in (best_weight, best_bias):
        axes.axhline(value, color=MUTED, linestyle="--", linewidth=1)
    axes.set_ylim(-0.5, max(run.weights) + 4)
    style_axes(axes, f"6 · Why rate {learning_rate:g} makes only the weight wobble",
               "training step", "value")
    note(axes,
         f"one rate meets two curvatures.  the weight's gradient is scaled by\n"
         f"mean(x²) = {curvature:g}, the bias's by 1.  so each step moves the weight\n"
         f"{learning_rate:g} × {curvature:g} = {learning_rate * curvature:g} of the way to the bottom — past 1, so it\n"
         f"overshoots and lands on the far side — and the bias {learning_rate:g} of the\n"
         f"way, which just settles.  (past 2 the weight would diverge.)",
         0.03, 0.97, va="top")
    legend(axes, "center right")


def show_training_report(
    x: np.ndarray,
    targets: np.ndarray,
    forward: Callable[[np.ndarray, float, float], np.ndarray],
    mean_half_squared_error: Callable[[np.ndarray, np.ndarray], float],
    loss_and_gradients: Callable[[np.ndarray, np.ndarray, float, float], tuple],
    train: Callable[..., tuple],
    wobbling_rate: float = 0.9,
    **options,
) -> Path:
    """Draw six views of one training run.  Hand it the four functions you wrote."""

    def loss_at(weight: float, bias: float) -> float:
        return float(mean_half_squared_error(forward(x, weight, bias), targets))

    def run_at_rate(learning_rate: float | None = None) -> Run:
        # train records each quantity *before* its update, so the final weight
        # and bias are missing from the history.  Append them: without them,
        # every graph would stop one step short of the answer.
        settings = dict(options)
        if learning_rate is not None:
            settings["learning_rate"] = learning_rate
        final_weight, final_bias, history = train(x, targets, **settings)
        return Run(
            steps=[item.step for item in history] + [len(history)],
            weights=[item.weight for item in history] + [final_weight],
            biases=[item.bias for item in history] + [final_bias],
            losses=[item.loss for item in history] + [loss_at(final_weight, final_bias)],
        )

    # The line that actually minimises the loss, worked out from the data rather
    # than assumed — so these graphs stay honest if you add noise to the targets.
    centred_x = x - x.mean()
    best_weight = float((centred_x * (targets - targets.mean())).sum() / (centred_x**2).sum())
    best_bias = float(targets.mean() - best_weight * x.mean())

    run = run_at_rate()
    _, gradient_weight, gradient_bias = loss_and_gradients(x, targets, run.weights[0], run.biases[0])
    first_errors = forward(x, run.weights[0], run.biases[0]) - targets

    figure, axes = plt.subplots(2, 3, figsize=(16.5, 9), facecolor=SURFACE)
    figure.suptitle(
        f"Two knobs learning a line from {len(x)} examples at once",
        color=INK, fontsize=14, fontweight="bold", x=0.012, ha="left",
    )
    plot_loss(axes[0][0], run)
    plot_fitted_lines(axes[0][1], run, x, targets)
    plot_loss_surface(axes[0][2], run, loss_at, best_weight, best_bias)
    plot_parameters(axes[1][0], run, best_weight, best_bias)
    plot_batch_votes(axes[1][1], x, first_errors, float(gradient_weight), float(gradient_bias))
    plot_two_curvatures(axes[1][2], run_at_rate(wobbling_rate), float(np.mean(x**2)),
                        wobbling_rate, best_weight, best_bias)
    figure.tight_layout(rect=(0, 0, 1, 0.955))

    return finish(figure, Path(__file__).with_name("solution2_training.png"))
