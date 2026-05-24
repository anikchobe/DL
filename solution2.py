from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from solution2_viz import show_training_report


@dataclass(frozen=True)
class RegressionStep:
    step: int
    weight: float
    bias: float
    loss: float


def forward(x: np.ndarray, weight: float, bias: float) -> np.ndarray:
    # TODO: vectorized predictions for every example.
    return x * weight + bias


def mean_half_squared_error(predictions: np.ndarray, targets: np.ndarray) -> float:
    # TODO: average 0.5 * error^2 over the batch.
    return 0.5 * np.mean((predictions - targets)**2)


def loss_and_gradients(
    x: np.ndarray, targets: np.ndarray, weight: float, bias: float
) -> tuple[float, float, float]:
    # TODO: return loss, dL/dweight, and dL/dbias.
    predictions = forward(x, weight, bias)
    errors = predictions -  targets
    loss = mean_half_squared_error(predictions, targets)
    gradient_weight = np.mean(errors * x)
    gradient_bias = np.mean(errors)
    return loss, gradient_weight, gradient_bias

# at learning_rate = 0.9, why does weight oscillate while bias converges?

def train(
    x: np.ndarray,
    targets: np.ndarray,
    *,
    initial_weight: float = 0.0,
    initial_bias: float = 0.0,
    learning_rate: float = 0.6,
    steps: int = 20,
) -> tuple[float, float, list[RegressionStep]]:
    # TODO: repeat gradient descent updates.
    weight = initial_weight
    bias = initial_bias
    history: list[RegressionStep] = []
    for step in range(steps):
        loss, gradient_weight, gradient_bias = loss_and_gradients(x, targets, weight, bias)
        history.append(RegressionStep(step, weight, bias, loss))
        weight -= learning_rate * gradient_weight
        bias -= learning_rate * gradient_bias
    return weight, bias, history

x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
y = 3.0 * x + 2.0
learned_w, learned_b, history = train(x, y)
print(f'w = {learned_w}, b = {learned_b}')
print("step | weight     | bias | loss       ")
for item in history:
    print(f"{item.step:>4} | {item.weight:>10.6f} | {item.bias:>10.6f} | {item.loss:>10.6f}")

# The same run as a picture.  All of the matplotlib code lives in
# solution2_viz.py; it only needs the four functions above.
figure_path = show_training_report(x, y, forward, mean_half_squared_error, loss_and_gradients, train)
print(f"\nSaved the training graphs to {figure_path}")


# Derivation for gradients:
# predictions = forward(x, weight, bias)   # 5 predictions
# errors = predictions - targets           # 5 errors, still a vector
# loss = float(0.5 * np.mean(errors**2))   # collapse to 1 scalar
# gradient_weight = float(np.mean(errors * x))   # how should the slope change?
# gradient_bias   = float(np.mean(errors))       # how should the intercept change?

# Two returns instead of Stage 1's one, because there are now two knobs. Each answers the same question independently: "if I nudge this knob up a little, what happens to the loss?"
# The derivation
# loss = (1/N) · Σᵢ 0.5·(w·xᵢ + b − tᵢ)²        let eᵢ = w·xᵢ + b − tᵢ
# ∂loss/∂w = (1/N) · Σᵢ eᵢ · xᵢ  = mean(errors * x)     ← the xᵢ comes from d(w·xᵢ)/dw
# ∂loss/∂b = (1/N) · Σᵢ eᵢ · 1   = mean(errors)         ← d(b)/db = 1
# The 1/N in both gradients isn't an extra design choice — line 48's comment says it: the mean in the loss is where it comes from. Mean loss ⇒ mean gradient, automatically.


# why * x for weight and nothing for bias

# This is the payoff from Stage 1's leverage idea. A parameter's gradient is scaled by how much influence it has on the prediction:
# - weight is multiplied by x, so its influence varies per example. An example with x = 2 is twice as sensitive to a slope change as one with x = 1.
# Big-x examples get a louder vote in gradient_weight.
# - bias is added to every prediction equally — its influence is always exactly 1. So every example votes with equal weight, and no scaling factor appears.
# Great concrete check in this data: x[2] = 0. That example still has an error (-1.7), but errors * x at that position is -0.0 — it contributes nothing
# to the weight gradient, and correctly so, since changing the slope can't move a point sitting on the y-axis. It does contribute fully to the bias gradient. 

# What batching actually means
# From problem2.md: "Learn how a shared parameter receives gradient contributions from the whole batch."
# weight is one knob serving all 5 examples. Each example has its own opinion about which way to turn it — and np.mean is how those 5 opinions become one decision.
# At weight=-0.7, bias=0.3 the per-example votes are:
# errors * x  =  [-11.4, -2.0, -0.0, -5.4, -18.2]     →  mean = -7.4
# Here they all point the same way, so the consensus is clear. In general they disagree — some examples want the slope up, others down — and the gradient is the compromise.
# That's the entire idea of batch training: no single example gets to yank the parameters around; you move in the direction that helps on average.

# One subtlety: both gradients are computed before either parameter moves. If you updated weight first and then recomputed the bias gradient at the new weight,
# you'd be measuring a different surface than the one you decided to descend.
# Both must be measured at the same point, then applied together. It's easy to get wrong during implementation, and it stays true for all 100,000 parameters later.

# The weight converges faster than the bias. Under default parameter, by step 10 the slope is at 2.68 (89% of the way) while the intercept is only at 1.30 (65%).
# That's the * x factor again. x values up to 2 make the weight gradient larger, so it moves faster under the same learning rate.
# Different parameters learning at different speeds is normal, and it's the reason later problems care about input scaling.

# From question above:
# at learning_rate = 0.9, why does weight oscillate while bias converges?

# the weight sits in a steeper direction of the loss surface than the bias — exactly 2× steeper — so 0.9 is past the oscillation threshold for the weight
# but comfortably under it for the bias. Here's the derivation to hand the student.
# Change coordinates to "how wrong is each parameter"
# Let u = weight − 3 and v = bias − 2 (the remaining error in each parameter; both should reach 0). Then each per-example error becomes:
# eᵢ = w·xᵢ + b − tᵢ = (w−3)·xᵢ + (b−2) = u·xᵢ + v
# Substitute into the two gradient formulas from the last stage:
# grad_w = mean(eᵢ·xᵢ) = u·mean(x²) + v·mean(x)
# grad_b = mean(eᵢ)     = u·mean(x)  + v
# Now plug in the actual data, x = [-2,-1,0,1,2]:
# mean(x)  = 0        ← the data is centered
# mean(x²) = 2
# mean(x) = 0 kills both cross terms, and the two parameters completely decouple:
# grad_w = 2u          grad_b = v
# The update becomes a multiplication
# Substituting into weight -= lr * grad_w and bias -= lr * grad_b:
# u ← u − lr·2u = u·(1 − 2·lr)
# v ← v − lr·v  = v·(1 − 1·lr)
# Each parameter's error just gets multiplied by a constant every step. That constant is the entire story. At lr = 0.9:
# ┌────────┬────────────────────┬───────────────────────────────────────────────────────────────────────────────────────┐
# │        │     multiplier     │                                       behavior                                        │
# ├────────┼────────────────────┼───────────────────────────────────────────────────────────────────────────────────────┤
# │ weight │ 1 − 2(0.9) =  −0.8 │ negative → sign flips every step = oscillation; |−0.8| < 1 → amplitude decays, slowly │
# ├────────┼────────────────────┼───────────────────────────────────────────────────────────────────────────────────────┤
# │ bias   │ 1 − 1(0.9) =  +0.1 │ positive → no flip; shrinks 10× per step                                              │
# └────────┴────────────────────┴───────────────────────────────────────────────────────────────────────────────────────┘
# That's the answer. The oscillation is the negative sign; convergence is the magnitude being under 1. The weight has both, so it oscillates while converging.
# The trace shows the multiplier is exactly −0.8 at every step:shows the multiplier is exactly −0.8 at every step:
# step   weight     bias       w−3         b−2         ratio of successive (w−3)
#    0   0.000000   0.000000   −3.000000   −2.000000        —
#    1   5.400000   1.800000   +2.400000   −0.200000     −0.8000
#    2   1.080000   1.980000   −1.920000   −0.020000     −0.8000
#    3   4.536000   1.998000   +1.536000   −0.002000     −0.8000
#    4   1.771200   1.999800   −1.228800   −0.000200     −0.8000
# Weight alternates above/below 3 forever, shedding only 20% per step. Bias is already correct to 4 decimals by step 4 and never overshoots once.
# Where the 2 comes from — the part worth internalizing
# A parameter's steepness is the mean squared value of whatever it multiplies.
# - weight multiplies x → its curvature is mean(x²) = 2
# - bias multiplies the implicit constant 1 → its curvature is mean(1²) = 1
# So the loss surface is a valley that's twice as steep along the weight axis as along the bias axis, and one shared learning_rate has to work for both.
# Formally that's the Hessian [[mean(x²), mean(x)], [mean(x), 1]] = [[2,0],[0,1]] — diagonal precisely because the data is centered.

# This is also the honest version of what I said last time about the weight converging faster at lr=0.1. I attributed it to the * x leverage factor;
# mean(x²) = 2 is that factor made exact. The same 2 that made the weight learn faster at 0.1 is what makes it oscillate at 0.9 — steeper means faster,
# right up until it means unstable.

# The general rule
# Convergence needs |1 − lr·c| < 1, i.e. 0 < lr < 2/c, and monotone (non-oscillating) behavior needs lr < 1/c:
# ┌─────────────┬──────────┬──────────────────────────┬──────────┐
# │ curvature c │ monotone │ oscillates but converges │ diverges │
# ├─────────────┼──────────┼──────────────────────────┼──────────┤
# │ weight, c=2 │ lr < 0.5 │ 0.5 < lr < 1.0           │ lr > 1.0 │
# ├─────────────┼──────────┼──────────────────────────┼──────────┤
# │ bias, c=1   │ lr < 1.0 │ 1.0 < lr < 2.0           │ lr > 2.0 │
# └─────────────┴──────────┴──────────────────────────┴──────────┘
# 0.9 lands in the weight's oscillating band and the bias's monotone band. Exactly the observed behavior.
# Two demos that make the table concrete — both verified:
# lr=0.5   weight per step:  0.0, 3.0, 3.0, 3.0, ...     ← multiplier 0: correct in ONE step
# lr=1.0   weight per step:  0.0, 6.0, 0.0, 6.0, ...     ← multiplier −1: oscillates forever, never decays
# lr=1.05  weight per step:  0.0, 6.3, −0.63, 6.993, −1.3923, 7.8315   ← diverges
# lr = 1/c is the perfect rate — it lands exactly on the answer in a single step. But it's 0.5 for the weight and 1.0 for the bias, and you only get one learning rate.
# Pick 0.5 and the weight is instantly perfect while the bias crawls; pick 1.0 and the bias is instantly perfect while the weight ping-pongs forever.
# That tension has a name — ill-conditioning — and here the condition number is only 2/1 = 2, about as benign as it gets. It's still enough to produce the  question above

# Why this is the most useful question they could have asked

# It generalizes directly to everything after this stage:
# Centering the inputs is what made the analysis easy. mean(x) = 0 is what decoupled the two parameters. With uncentered data the cross terms survive and the parameters
# fight each other. Same lr=0.9, same line to learn, just shifted to x = [10,11,12,13,14]:

# weight:  0.0, 415.8, −54173.9, 7112782.4, −933820806.4  → explodes
# bias:    0.0,  34.2,  −4453.0,  584666.8,  −76759548.7  → explodes

# The bias diverges too now, even though its own curvature is still 1 — coupling drags it along. This is the concrete reason MNIST pixels get divided by 255 in Stage 6
# rather than fed in as 0–255. With raw pixels mean(x²) is in the tens of thousands, and the stable lr ceiling collapses to something like 10⁻⁴.

# And it's the motivation for optimizers. Momentum, RMSProp, and Adam all exist to cope with different parameters wanting different learning rates
# which is precisely what this 2-vs-1 is, in miniature.

# Good experiment to end on: sweep lr over 0.1, 0.4, 0.5, 0.6, 0.9, 0.99, 1.0, 1.01 and predict the multiplier 1 − 2·lr before running each one.
