"""Draw the training run from solution1.py.

Nothing here is part of the exercise — the maths all lives in solution1.py, and
this file only reads its output.  It is kept separate so that solution1.py stays
short enough to hold in your head.

Run it with:  python3 solution1.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt

from viz_style import (
    AQUA, AXIS, BLUE, BLUE_RAMP_4, INK, MUTED, ORANGE, SURFACE,
    draw_line, finish, legend, note, style_axes,
)

# One colour, one meaning, in every panel: blue is the loss, orange is the
# weight, aqua is the prediction.
LOSS_COLOR = BLUE
WEIGHT_COLOR = ORANGE
PREDICTION_COLOR = AQUA

# Learning rates are *ordered*, so they get one hue from light to dark instead
# of four unrelated colours: darker means a bigger step.
LEARNING_RATE_COLORS = BLUE_RAMP_4


@dataclass(frozen=True)
class Run:
    """One training run, with each quantity as a list ready to plot."""

    steps: list[int]
    weights: list[float]
    predictions: list[float]
    losses: list[float]
    gradients: list[float]


def plot_loss(axes, run: Run) -> None:
    """Panel 1 — the headline every training loop is judged by."""
    draw_line(axes, run.steps, run.losses, LOSS_COLOR)
    style_axes(axes, "1 · The loss falls", "training step", "loss")
    axes.annotate(
        f"starts at {run.losses[0]:.0f}\nweight 0 predicts 0, we wanted 12",
        xy=(run.steps[0], run.losses[0]), xytext=(14, -16),
        textcoords="offset points", color=MUTED, fontsize=8.5,
    )
    axes.annotate(
        f"ends at {run.losses[-1]:.1e}",
        xy=(run.steps[-1], run.losses[-1]), xytext=(-6, 18),
        textcoords="offset points", color=MUTED, fontsize=8.5, ha="right",
    )


def plot_loss_log(axes, run: Run) -> None:
    """Panel 2 — the same numbers on a log axis.  A straight line is a result."""
    visible = [(step, loss) for step, loss in zip(run.steps, run.losses) if loss > 0]
    draw_line(axes, [step for step, _ in visible], [loss for _, loss in visible], LOSS_COLOR)
    axes.set_yscale("log")
    style_axes(axes, "2 · The same loss, log axis", "training step", "loss (log scale)")
    shrink = run.losses[1] / run.losses[0]
    note(axes,
         "a straight line here means the loss shrinks by the\n"
         f"same factor ({shrink:.2f}x) every step.  Panel 1 flattens\n"
         "out; it is not stalling, it is running out of room.",
         0.04, 0.06)


def plot_loss_landscape(
    axes, run: Run, loss_at_weight: Callable[[float], float], best_weight: float
) -> None:
    """Panel 3 — why any of this works: the loss is a bowl and we walk downhill."""
    grid = [-1 + index * 0.05 for index in range(201)]
    axes.plot(
        grid, [loss_at_weight(weight) for weight in grid],
        color=LOSS_COLOR, linewidth=2, label="loss at every possible weight",
    )
    draw_line(axes, run.weights, run.losses, WEIGHT_COLOR, label="weights training visited")
    axes.axvline(best_weight, color=MUTED, linestyle="--", linewidth=1)
    axes.set_ylim(-6, 128)  # headroom, so the notes never sit on the curve
    style_axes(axes, "3 · Training walks downhill on a bowl", "weight", "loss")
    axes.annotate(
        f"bottom of the bowl:\nweight {best_weight:g}, loss 0",
        xy=(best_weight, 0), xytext=(12, 42),
        textcoords="offset points", color=MUTED, fontsize=8.5,
    )
    axes.annotate(
        "starts here, takes big steps\nwhile the slope is steep",
        xy=(run.weights[0], run.losses[0]), xytext=(40, 16),
        textcoords="offset points", color=WEIGHT_COLOR, fontsize=8.5,
    )
    legend(axes, "upper right")


def plot_convergence(axes, run: Run, best_weight: float, target: float) -> None:
    """Panel 4 — the two numbers you would actually report."""
    draw_line(axes, run.steps, run.weights, WEIGHT_COLOR, label="weight")
    draw_line(axes, run.steps, run.predictions, PREDICTION_COLOR, label="prediction")
    for value, text in ((best_weight, f"the weight we want, {best_weight:g}"),
                        (target, f"the target, {target:g}")):
        axes.axhline(value, color=MUTED, linestyle="--", linewidth=1)
        axes.annotate(
            text, xy=(0, value), xytext=(4, 6),
            textcoords="offset points", color=MUTED, fontsize=8.5,
        )
    # Label both ends directly, so the two series are never colour-only.
    for values, color, name in ((run.weights, WEIGHT_COLOR, "weight"),
                                (run.predictions, PREDICTION_COLOR, "prediction")):
        axes.annotate(
            f"{name} {values[-1]:.3f}", xy=(run.steps[-1], values[-1]), xytext=(-6, -18),
            textcoords="offset points", color=color, fontsize=8.5, ha="right",
        )
    style_axes(axes, "4 · Weight and prediction reach their targets", "training step", "value")
    legend(axes, "center right")


def plot_gradient(axes, run: Run) -> None:
    """Panel 5 — the gradient fades, so the steps shrink by themselves."""
    axes.bar(run.steps, run.gradients, color=LOSS_COLOR, width=0.6)
    axes.axhline(0, color=AXIS, linewidth=1)
    style_axes(axes, "5 · The gradient fades to zero", "training step", "gradient")
    note(axes,
         f"the gradient is negative ({run.gradients[0]:.0f} at the start), and\n"
         "`weight -= learning_rate * gradient` subtracts a negative,\n"
         "so the weight goes up.  Near the bottom of the bowl the\n"
         "gradient is tiny, so nobody has to tell the steps to shrink.",
         0.97, 0.06, ha="right")


def plot_learning_rates(
    axes, run_at_rate: Callable[[float], Run], best_weight: float,
    rates=(0.001, 0.01, 0.05, 0.2), diverging_rate=1.0,
) -> None:
    """Panel 6 — the one knob you have to choose, and what it does."""
    notes = {0.001: "too slow", 0.01: "slow", 0.05: "just right", 0.2: "overshoots, still lands"}
    for rate, color in zip(rates, LEARNING_RATE_COLORS):
        run = run_at_rate(rate)
        draw_line(axes, run.steps, run.weights, color, label=f"rate {rate} — {notes[rate]}", marker="")
    # Labelled in the legend rather than on the line: every spot near weight 4
    # already has a curve running through it.
    axes.axhline(best_weight, color=MUTED, linestyle="--", linewidth=1,
                 label=f"the weight we want, {best_weight:g}")
    axes.set_ylim(-0.6, 12)  # headroom for the note and the legend
    style_axes(axes, "6 · The learning rate is the knob", "training step", "weight")

    # A rate of 1.0 leaves this chart on its first step, so name it rather than
    # silently drop it.
    path = " → ".join(f"{weight:g}" for weight in run_at_rate(diverging_rate).weights[:3])
    note(axes,
         f"learning rate {diverging_rate:g} diverges — one step\n"
         f"takes it off this chart:\n{path} → …",
         0.03, 0.97, va="top")
    legend(axes, "upper right")


def show_training_report(
    x: float,
    target: float,
    forward: Callable[[float, float], float],
    half_squared_error: Callable[[float, float], float],
    backward: Callable[[float, float, float], float],
    train_weight: Callable[..., tuple],
    **options,
) -> Path:
    """Draw six views of one training run.  Hand it the four functions you wrote."""

    def loss_at_weight(weight: float) -> float:
        return half_squared_error(forward(x, weight), target)

    def run_at_rate(learning_rate: float | None = None) -> Run:
        # train_weight records each quantity *before* its update, so the final
        # weight is missing from the history.  Append it: without it, every
        # graph would stop one step short of the answer.
        settings = dict(options)
        if learning_rate is not None:
            settings["learning_rate"] = learning_rate
        final_weight, history = train_weight(x, target, **settings)
        return Run(
            steps=[item.step for item in history] + [len(history)],
            weights=[item.weight for item in history] + [final_weight],
            predictions=[item.prediction for item in history] + [forward(x, final_weight)],
            losses=[item.loss for item in history] + [loss_at_weight(final_weight)],
            gradients=[item.gradient for item in history] + [backward(x, final_weight, target)],
        )

    run = run_at_rate()
    best_weight = target / x

    figure, axes = plt.subplots(2, 3, figsize=(16.5, 9), facecolor=SURFACE)
    figure.suptitle(
        f"One weight learning that {x:g} × weight should be {target:g}",
        color=INK, fontsize=14, fontweight="bold", x=0.012, ha="left",
    )
    plot_loss(axes[0][0], run)
    plot_loss_log(axes[0][1], run)
    plot_loss_landscape(axes[0][2], run, loss_at_weight, best_weight)
    plot_convergence(axes[1][0], run, best_weight, target)
    plot_gradient(axes[1][1], run)
    plot_learning_rates(axes[1][2], run_at_rate, best_weight)
    figure.tight_layout(rect=(0, 0, 1, 0.955))

    return finish(figure, Path(__file__).with_name("solution1_training.png"))
