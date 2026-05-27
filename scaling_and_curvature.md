# Why the loss explodes when you change `x` — solution2

Three observations, one cause.

```text
x = [10, 11, 12, 13, 14]     -> loss explodes
x = [-20, -10, 0, 10, 20]    -> loss explodes, even though it is centered on 0
x / 20                       -> converges, but needs far more steps
```

All three are the same fact seen from three angles: **the learning rate you can
use is decided by the data, not by the model.** `lr = 0.1` was never a property
of the problem. It was a property of `x = [-2, -1, 0, 1, 2]`.

Everything below is verified numerically against `solution2.py`'s exact update
rule.

---

## 1. The one formula

Keep the change of coordinates from the `lr = 0.9` analysis. Let

```text
u = weight - 3      v = bias - 2        (how wrong each parameter is; both -> 0)
```

Then each per-example error is `e_i = u*x_i + v`, and the two gradients are

```text
grad_w = u*mean(x^2) + v*mean(x)
grad_b = u*mean(x)   + v
```

which is a matrix times a vector. So one gradient-descent step is a **linear
map**:

```text
[u, v]  <-  (I - lr*H) [u, v]           H = [[ mean(x^2), mean(x) ],
                                             [ mean(x),      1     ]]
```

`H` is the curvature of the loss surface (the Hessian). Its two eigenvalues
`lam` are the only things that matter. Along each eigen-direction the error is
simply **multiplied by `1 - lr*lam` every step**:

```text
|1 - lr*lam| < 1     converges          <=>  lr < 2/lam
 1 - lr*lam  < 0     sign flips each step (oscillates)
|1 - lr*lam| > 1     explodes           <=>  lr > 2/lam
```

Two consequences, and every question you asked is one of them:

```text
STABILITY is set by the LARGEST eigenvalue:   lr < 2/lam_max
SPEED     is set by the SMALLEST eigenvalue:  slow direction shrinks by (1 - lr*lam_min)
```

The ratio `kappa = lam_max / lam_min` (the *condition number*) says whether a
single `lr` can serve both parameters at once. `kappa = 1` is perfect. Large
`kappa` means whatever you pick is too big for one parameter or too small for
the other.

### The decomposition that answers everything

```text
mean(x^2) = mean(x)^2 + var(x)
            ^^^^^^^^^   ^^^^^^
            how far the  how spread
            data sits    out the data
            from zero    is
```

The weight's curvature is `mean(x^2)`, and it has **two independent ways of
being large**. Your two exploding cases are one of each.

---

## 2. Case 1 — `x = [10, 11, 12, 13, 14]`: large because it is *offset*

```text
mean(x)   = 12          mean(x)^2 = 144
var(x)    = 2           mean(x^2) = 144 + 2 = 146

H = [[146, 12],
     [ 12,  1]]         eigenvalues: 146.99  and  0.0136      kappa = 10,800
```

Stability needs `lr < 2/146.99 = 0.0136`. You used `0.1`, which is **7x past the
cliff**. The multiplier is

```text
1 - 0.1 * 146.99 = -13.7
```

Error multiplied by −13.7 every step; loss, being quadratic, multiplied by
`13.7^2 = 188` every step. Verified trace:

```text
step  0   loss 7.31e+02
step  1   loss 1.37e+05
step  2   loss 2.57e+07
step  3   loss 4.83e+09
step 19   loss 1.14e+46
```

**Read where the 146 comes from.** The data spans only 4 units, so `var(x) = 2`
— exactly the same spread as the original `[-2..2]`. The other **144 comes
purely from the offset.** 98.6% of the curvature is caused by the data sitting
12 units away from the origin, and none of it by the shape of the data.

Why the offset hurts, intuitively: `bias` is the value of the line *at x = 0*,
and there is no data anywhere near `x = 0`. The model is forced to extrapolate
back 12 units to place the intercept. That has two effects:

- A small change in the slope swings the predictions at `x = 12` by **12x** as
  much. Hence the huge `lam_max`.
- Slope and intercept can no longer move independently. To keep the line running
  through the data cloud, raising `w` by `d` demands lowering `b` by about
  `12*d`. That is the off-diagonal `12` in `H`, and it is why **the bias
  explodes too** even though its own curvature is still just `1`. It is not
  unstable on its own; it is dragged.

---

## 3. Case 2 — `x = [-20, -10, 0, 10, 20]`: large because it is *spread out*

This is the important one, because it is the case that surprised you.

```text
mean(x)   = 0           mean(x)^2 = 0        <- centering worked
var(x)    = 200         mean(x^2) = 0 + 200 = 200

H = [[200, 0],
     [  0, 1]]          eigenvalues: 200 and 1          kappa = 200
```

Centering did exactly what centering does, and it did it perfectly:

- `mean(x) = 0` zeroed the off-diagonal, so `H` is **diagonal** — the two
  parameters are fully decoupled again, no more tug-of-war.
- `kappa` fell from 10,800 to 200.

What centering did **not** do is make the curvature *small*. It removed the
`mean(x)^2 = 144` term. It cannot touch `var(x)`, and here `var(x) = 200` is
larger than the entire curvature of case 1.

```text
weight multiplier:  1 - 0.1*200 = -19      explodes  (loss x 361 per step)
bias   multiplier:  1 - 0.1*  1 = +0.9     converges perfectly
```

And that is precisely what the trace shows — look at the bias column:

```text
step   weight        bias          loss
  0    0.0000e+00    0.0000e+00    9.02e+02
  1    6.0000e+01    2.0000e-01    3.25e+05
  2   -1.0800e+03    3.8000e-01    1.17e+08
  3    2.0580e+04    5.4200e-01    4.23e+10
  4   -3.9096e+05    6.8780e-01    1.53e+13
```

The bias marches calmly `0.2, 0.38, 0.542, 0.688` toward 2 — identical to the
well-behaved run — while the weight explodes beside it. They are decoupled, so
the bias genuinely does not care. The loss explodes anyway because the loss sees
both.

> **The lesson: "centered around 0" and "small" are two different properties, and
> stability needs the second one.** The threshold is `mean(x^2)`, and
> `mean(x^2) = mean(x)^2 + var(x)`. Centering kills the first term only.

Geometric picture: the loss surface is a long narrow valley. Centering **rotates**
the valley so its axes line up with the `weight` and `bias` axes. It does not
make the valley any shallower. You still fall out of it.

---

## 4. Case 3 — divide by 20: why it converges but crawls

```text
x = [-1, -0.5, 0, 0.5, 1]
mean(x^2) = 0.5         H = [[0.5, 0], [0, 1]]      eigenvalues 0.5 and 1
kappa = 2               stability ceiling: lr < 2/1 = 2.0
```

Note the surprise: this is the **best-conditioned** of all your cases —
`kappa = 2`, better than the original `[-2..2]`. Nothing is ill-conditioned any
more. So why is it slow?

Because `lr = 0.1` is now **20x below** the ceiling of 2.0. You fixed the data
and left the learning rate where it was.

```text
weight multiplier: 1 - 0.1*0.5 = 0.95      <- removes only 5% of the error per step
bias   multiplier: 1 - 0.1*1.0 = 0.90
```

Steps needed for `|w - 3| < 1e-4`, predicted and measured:

```text
predicted:  ln(1e-4 / 3) / ln(0.95) = 200.98
measured:   first within tolerance at step 201        <- exact
```

So it is not a leftover conditioning problem. **Scaling the inputs down by `s`
divides the weight's curvature by `s^2`, which multiplies the usable learning
rate by `s^2`.** You divided by 20, so the learning rate could have gone up by
up to 400x. Leaving it at 0.1 is what costs the steps.

**The fix is to raise `lr`, not `steps`.** Measured:

```text
lr = 0.1    ->  201 steps
lr = 0.5    ->   36 steps
lr = 1.0    ->   15 steps
lr = 4/3    ->   10 steps        <- optimal: lr = 2/(lam_max + lam_min) = 2/1.5
```

**One thing to check in your code:** did you regenerate the targets as
`y = 3*x_new + 2`, or keep the old `y` computed from the original `x`? If you
kept the old targets, the true weight became **60**, not 3 — you divided the
inputs by 20, so the slope must be 20x larger to compensate. Same curvature and
same multiplier, but now 20x further to travel, costing about 60 extra steps.
Both versions converge; they just are not the same problem.

---

## 5. The endpoint: standardize, and all three become identical

```text
mean(x^2) = mean(x)^2 + var(x)
             ^                ^
             subtract the mean |
             kills this term   divide by the std
                               kills this one (sets var = 1)
```

Do both — `z = (x - mean(x)) / std(x)` — and `mean(z^2) = 1`, which happens to be
exactly the bias's curvature. So:

```text
H = [[1, 0],        kappa = 1        every direction equally steep
     [0, 1]] = I
```

Now a single learning rate is simultaneously optimal for both parameters, and
`lr = 1` gives multiplier `1 - 1*1 = 0`: the exact answer in **one step**.
Verified on all three of your datasets:

```text
x = [-2, -1, 0, 1, 2]      -> standardize, lr=1.0, 1 step -> w=3.0000000000  b=2.0000000000
x = [-20, -10, 0, 10, 20]  -> standardize, lr=1.0, 1 step -> w=3.0000000000  b=2.0000000000
x = [10, 11, 12, 13, 14]   -> standardize, lr=1.0, 1 step -> w=3.0000000000  b=2.0000000000
```

That is the real punchline. The three datasets that behaved so differently are
**the same problem wearing different coordinates**, and standardizing makes them
literally identical. The wild differences in behavior were never about the data;
they were about the mismatch between one fixed `lr` and three different
curvatures.

---

## 6. The whole thing in one table

All values verified numerically.

| `x` | `mean(x)^2` | `var(x)` | `mean(x^2)` | `lam_max` | `lam_min` | `kappa` | max stable `lr` | behavior at `lr = 0.1` |
|---|---|---|---|---|---|---|---|---|
| `[-2..2]` (course) | 0 | 2 | 2 | 2 | 1 | 2 | 1.0 | converges, 20 steps |
| `[10..14]` | 144 | 2 | 146 | 147.0 | 0.0136 | 10,800 | 0.0136 | x(−13.7)/step, explodes |
| `[-20..20]` | 0 | 200 | 200 | 200 | 1 | 200 | 0.010 | x(−19)/step, explodes |
| `/20` = `[-1..1]` | 0 | 0.5 | 0.5 | 1 | 0.5 | 2 | 2.0 | x0.95/step, 201 steps |
| standardized | 0 | 1 | 1 | 1 | 1 | **1** | 2.0 | 1 step at `lr = 1` |

Read the two exploding rows against each other: they have almost nothing in
common. One has tiny variance and a big offset; the other has zero offset and
huge variance. Only `mean(x^2)` explains both.

---

## 7. Why "explodes" means *exponentially*

It is a feedback loop, and each turn of it is the same multiplication:

```text
overshoot  ->  error is larger  ->  gradient is proportionally larger
           ->  next step overshoots by even more  ->  ...
```

Because the gradient is *linear* in the error, the error gets multiplied by the
same constant every single step. Constant multiplication is exponential growth
by definition. That is why the loss reaches `1e46` in 20 steps and `inf`/`nan`
shortly after — nothing "goes wrong" partway through; the very first step already
contains the whole future.

**Diagnostic worth keeping.** Print `weight - 3` each step and take the ratio of
consecutive values. It is a constant, and it equals `1 - lr*lam`:

- `|ratio| > 1` -> learning rate too large, it will explode
- `ratio < 0` -> oscillating
- `ratio` near 1 -> learning rate too small, it will crawl

From a measured ratio you can solve for the curvature: `lam = (1 - ratio)/lr`.
This works on any model, including ones far too big to write a Hessian for.

---

## 8. Why this matters beyond this toy

- **MNIST pixels are 0–255 and get divided by 255.** That is this exact fix.
  Scaling inputs by `1/255` divides the curvature by `255^2` and therefore
  raises the usable learning rate by `255^2 = 65,025`. Feed raw bytes in and the
  stable `lr` collapses to something you would never guess by hand.
- **You cannot always fix `kappa` by preprocessing.** In a deep network the
  curvature of a hidden layer depends on weights that are still changing, so no
  one-time rescale of the inputs settles it. That is why **Adam and RMSProp**
  exist: they give each parameter its own effective learning rate, which is
  exactly the `2-vs-1` problem you already met at `lr = 0.9`, at scale. And it is
  why **batch normalization** exists: re-standardizing *between* layers, for the
  same reason you standardize the inputs.
- Note the ordering: your `lr = 0.9` question was about `kappa = 2` — one
  parameter oscillating while the other converged. Case 1 here is `kappa = 10,800`.
  Same phenomenon, five orders of magnitude worse.

---

## 9. Three experiments

**1. Find the cliff.** For `x = [-20, -10, 0, 10, 20]`, predict the multiplier
`1 - lr*200` *before* running each of `lr = 0.001, 0.005, 0.0099, 0.01, 0.011`.
Then run. Verified answers:

```text
lr = 0.001   multiplier +0.800   converges,  9899 steps
lr = 0.005   multiplier  0.000   converges,  1976 steps   <- weight exact in 1 step
lr = 0.0099  multiplier -0.980   converges,   996 steps   (oscillating)
lr = 0.01    multiplier -1.000   never converges, orbits forever
lr = 0.011   multiplier -1.200   diverges to inf
```

The cliff is at exactly `lr = 2/200 = 0.01`, right where the formula says.
Then explain the odd bit: at `lr = 0.005` the weight is *perfect after one step*,
yet the run still takes 1976. (Because the bias multiplier is `0.995`, and with
`kappa = 200` the binding constraint has flipped to the other parameter. That is
what a bad condition number costs you even when you are stable.)

**2. Even a perfect learning rate cannot rescue uncentered data.** For
`x = [10..14]`, use the best `lr` that exists, `2/(lam_max + lam_min) = 0.0136`.
Measured: **55,682 steps**. Compare with `kappa = 10,800` — the step count tracks
the condition number, not the learning rate. Conclusion: no learning rate fixes
bad conditioning. Only fixing the *data* does.

**3. Standardize, set `lr = 1`, run one step.** Then explain to me why one step
is enough, in terms of the multiplier.
