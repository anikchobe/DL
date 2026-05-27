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

And for the first half of the run that is exactly what the trace shows — look at
the bias column:

```text
step   weight        bias          loss
  0    0.0000e+00    0.0000e+00    9.02e+02
  1    6.0000e+01    2.0000e-01    3.25e+05
  2   -1.0800e+03    3.8000e-01    1.17e+08
  3    2.0580e+04    5.4200e-01    4.23e+10
  4   -3.9096e+05    6.8780e-01    1.53e+13
```

The bias marches calmly `0.2, 0.38, 0.542, 0.688` toward 2 — identical, digit for
digit, to the well-behaved `[-2..2]` run — while the weight explodes beside it.
The loss explodes anyway, because the loss sees both.

**But keep reading your own output past step 12, because it stops being true**,
and the reason it stops is worth more than the original point. Section 3a is
that story.

> **The lesson so far: "centered around 0" and "small" are two different
> properties, and stability needs the second one.** The threshold is `mean(x^2)`, and
> `mean(x^2) = mean(x)^2 + var(x)`. Centering kills the first term only.

Geometric picture: the loss surface is a long narrow valley. Centering **rotates**
the valley so its axes line up with the `weight` and `bias` axes. It does not
make the valley any shallower. You still fall out of it.

---

## 3a. Why the bias *also* goes off the rails after step 13

You spotted that the bias tracks toward 2 until about step 13 and then collapses:

```text
step 12   bias     1.431253
step 13   bias     1.751253      <- jumped 0.32 instead of the expected 0.06
step 14   bias    -8.488747      <- jumped -10.2
step 15   bias    -8.488747      <- didn't move at all
step 16   bias -2629.928747
step 17   bias -44572.968747
step 18   bias -44572.968747     <- frozen from here on
step 19   bias -44572.968747
```

The decoupling argument above is not wrong, but it is a statement about
**mathematics**, and `solution2.py` does not do mathematics — it does float64.
Those are the same thing right up until the weight gets large, and then they are
not.

### The mathematics says the bias converges forever

Re-run the identical update rule in exact rational arithmetic (`fractions.Fraction`,
no rounding anywhere) and the bias never misbehaves:

```text
step  0   bias 0.0000000000        predicted by multiplier 0.9:  2 - 2*0.9^0 = 0.0000000000
step 10   bias 1.3026431198                                      2 - 2*0.9^10 = 1.3026431198
step 20   bias 1.7568466908                                      2 - 2*0.9^20 = 1.7568466908
step 40   bias 1.9704382341                                      2 - 2*0.9^40 = 1.9704382341
step 60   bias 1.9964059794                                      2 - 2*0.9^60 = 1.9964059794
```

Ten-decimal agreement, forever, while the weight in the very same run explodes
past `1e24`. So the decoupling is real. What you are watching in float64 is not
the bias becoming unstable — **its multiplier is still 0.9 the whole time.** It
is the bias gradient being destroyed by rounding.

### What actually breaks: catastrophic cancellation

The bias gradient is `mean(errors)`, and with `mean(x) = 0` its true value is just
`v = bias - 2`, a number of size ~1. But look at what it is computed *from* at
step 13, where `w = 1.26e17`:

```text
errors = [-2.523e+18, -1.262e+18, -0.2487, +1.262e+18, +2.523e+18]
                                   ^^^^^^^
                            the only real signal
```

Five numbers of magnitude `1e18` that are supposed to cancel *exactly*, leaving a
residue of `-0.2487`. The trouble is the spacing between representable float64
numbers at that magnitude:

```text
ulp(2.523e18) = 512.0
```

The signal is destroyed twice over, at two different stages, and it is worth
separating them because they are different bugs that happen to co-occur.

**Damage 1 — in the elements, before any summing.** Each error should be
`err_i = u*x_i + v`. Forming it means adding `v` (size ~0.25) to `u*x_i` (size
`2.5e18`), where one unit in the last place is 512. So `v` is `0.05%` of an ulp
and simply vanishes. Check it directly — the outer pair should sum to `2v`:

```text
err[0] + err[4] = 0.0          but 2v = -0.4975
err[1] + err[3] = 0.0          but 2v = -0.4975
err[2]          = -0.2487      = v exactly, because x = 0 there
```

The bias survives at **exactly one of the five positions** — the point where
`x = 0`, which is the only one not swamped by `u*x_i`. (Print
`predictions - x*w` and you get `[0, 0, 1.75, 0, 0]`, which shows the same thing
one stage earlier.) So the true total `5v` has already decayed to `v` before
`np.mean` is even called. Even *perfect* summation now returns a gradient 5x too
small.

**Damage 2 — in the reduction.** `np.sum` adds those five numbers pairwise, which
means adding `-0.2487` into a partial sum of magnitude `3.78e18`:

```text
err[0] + err[1] = -3.7847685116031365e+18      ulp = 512.0
adding -0.2487 to that = 4.86e-04 ulps          -> rounds away completely
```

and the regrouping leaves behind a spurious `+512`. The result is
**summation-order dependent** — same five numbers, four different answers:

```text
np.sum   (pairwise)      =  512.0            <- what solution2.py actually gets
sum      (left to right) =   -0.24874697338282203
math.fsum (exact)        =   -0.24874697338288088
pairing symmetric terms  =   -0.24874697338288088
```

So `np.mean(errors)` returns `102.4 = 512/5` — *exactly one ulp divided by the
batch size* — where the true gradient is `-0.2487`. Wrong sign, wrong magnitude
by 400x, and containing no information whatsoever about the bias. That number is
pure quantization noise, and the bias is then updated with it.

If you take one thing from this: when a sum of large terms is supposed to cancel
down to a small one, the answer you get depends on the *order* the library
happens to add them in. That is the signature of catastrophic cancellation.

The decay is gradual and you can watch it happen. Comparing the computed
`mean(errors)` against the exact `v = bias - 2`:

```text
step   float64 mean(err)      exact v        ulp(max|pred|)
  6        -1.062882e+00     -1.062882         4.77e-07     agree to all digits
  9        -7.734375e-01     -0.774841         3.91e-03     3rd decimal wrong
 11        -6.000000e-01     -0.628747         1.00e+00     ulp reaches 1.0
 12        -3.200000e+00     -0.568747         1.60e+01     6x too big
 13        +1.024000e+02     -0.248747         5.12e+02     sign flipped, all noise
 15        +2.621440e+04    -10.488747         1.31e+05
```

Read the `ulp` column as the real clock of the run, and compare it against `|v|`,
the signal you are trying to measure — which is only ever about `1` and is
*shrinking* by 10% a step (multiplier `0.9`) while the ulp *grows* by about 20x a
step (multiplier `-19`). Digits start disappearing from the right as soon as the
ulp is within a few orders of magnitude of `v` — visible by step 7, obvious by
step 9 — and the two quantities cross between steps 10 and 11. By step 13 the ulp
is `512` against a signal of `0.2487`, a ratio of `2000:1`, and nothing is left.
That is precisely where you saw the trace break.

Note the cruel detail: the two quantities race in opposite directions. The better
the bias gets, the smaller `v` becomes and the *easier* it is to round away. The
bias is destroyed fastest exactly as it approaches the right answer.

### And why it then *freezes*

At steps 14, 17, 18 and 19 the bias does not move at all. The mechanism is the
one above, with the coin landing the other way. At step 17:

```text
err = [-3.288e+23, -1.644e+23, -44574.97, +1.644e+23, +3.288e+23]
```

The four large entries are *exactly* antisymmetric — `x` is symmetric and
round-to-nearest is sign-symmetric, so `fl(w*20)` and `fl(w*-20)` are exact
negatives — and they cancel to zero. The lone surviving term, `-44574.97`, is now
far below one ulp of the partial sums (`6.7e7`), so it contributes nothing at all.
`np.sum` returns exactly `0.0`, the gradient is zero, and the bias is frozen at
whatever the noisy steps happened to leave it at.

The difference between "spurious `+512`" and "exactly `0.0`" is nothing more than
which side of a rounding boundary the discarded term fell on. Both are noise; one
of them merely looks like convergence. **A parameter that stops changing is not
the same as a parameter that has converged** — here it is numerically dead, and
the loss beside it is `1e46`, which is how you can tell.

### The lesson, which is bigger than this bug

> **Decoupling is a property of the math, not of the computation.** Two parameters
> can be provably independent and still corrupt each other in floating point,
> because they share a reduction — here, the `np.mean` over the batch. A parameter
> that diverges poisons every quantity computed in the same sum as it, and it does
> so silently, producing plausible finite numbers rather than `nan`.

Note the contrast with case 1, and make sure you can state the difference:

```text
x = [10..14]     bias diverges because of COUPLING       (off-diagonal 12 in H)
                 -> still diverges in exact arithmetic, verified
x = [-20..20]    bias diverges because of CANCELLATION   (float64 only)
                 -> converges perfectly in exact arithmetic, verified
```

Same-looking symptom in the printout, two completely different causes. The only
way to tell them apart is to check whether the effect survives exact arithmetic —
which is worth remembering as a debugging technique in its own right.

Both, though, have the same *fix*, and it is the fix in section 5: don't let the
weight explode in the first place. Numerical catastrophes downstream of a
divergence are not worth debugging individually — the divergence is the bug.

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

Use it on the *first few* steps only. As section 3a showed, once the parameters
are large enough that float64 starts losing digits, the printed numbers stop
being the numbers the mathematics is talking about, and every ratio you measure
after that point is telling you about rounding rather than about curvature.
Diagnose a divergence at the beginning, where the trace is still trustworthy.

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

## 9. Four experiments

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

**4. Prove to yourself that the bias collapse is rounding and not dynamics.**
This is the experiment you effectively already started by reading past step 12.
For `x = [-20, -10, 0, 10, 20]`, add one line to the training loop:

```python
print(f"{step:>4}  mean(err)={np.mean(errors):>14.6e}   exact v={bias-2:>12.6f}"
      f"   ulp={np.spacing(np.max(np.abs(predictions))):>10.3e}")
```

Two things to find:

- The step at which `mean(err)` stops matching `bias - 2`. Then explain why that
  step is the one where `ulp` grows past the size of the numbers being cancelled.
- Re-run the *identical* loop with `weight` and `bias` as `fractions.Fraction`
  instead of floats (start them at `Fraction(0)` and use `Fraction(1, 10)` for the
  learning rate). The bias now converges to 2 forever. Whatever changes when you
  remove rounding was caused by rounding — that is the entire logic of the test,
  and it is how you settle this kind of question rather than arguing about it.

Then answer: the bias gradient is mathematically `bias - 2`, a number near `-0.5`.
Why does computing it as `mean(predictions - targets)` destroy it, when computing
it directly would not? (This is the same lesson as `logaddexp` in Stage 3 of the
course: *algebraically equal* and *numerically equal* are different claims, and
the difference is decided by what intermediate values you build on the way.)
