## Stage 4 — Ten competing classes: softmax and cross-entropy

### Goal

Build the probability and loss machinery needed for digit classification,
without training a model yet.  Every function in this stage is a pure function
of its arguments — no parameters, no updates, nothing to converge.  That is
deliberate.  Testing this machinery by hand now is what makes Stages 5, 8 and 11
debuggable, because from here on softmax and cross-entropy sit at the end of
every model you write and every gradient flows out of them.

Stage 3 answered one question: is this a 0 or a 1?  MNIST asks a harder one:
which of ten digits is this?  Read the change as a widening, not as a new idea:

```text
Stage 3                             Stage 4
-------                             -------
one logit per example               ten logits per example
sigmoid -> one probability          softmax -> ten probabilities summing to one
label 0 or 1                        label 0..9
loss = -log(p of true class)        loss = -log(p of true class)      unchanged
dloss/dlogit = p - y                dloss/dlogits = p - one_hot(y)    unchanged
```

The last two lines are neither a coincidence nor an approximation.  Stage 3's
formulas are the two-class special case of this stage's formulas, and Part H
asks you to show that in one line of algebra.

Shapes for the whole stage:

```text
logits:        (batch, 10)   any real numbers, one row per image
probabilities: (batch, 10)   positive, each row sums to 1
labels:        (batch,)      integers 0..9 — plain integers, not one-hot
predictions:   (batch,)      integers 0..9, the argmax of each row
```

Keep apart the same three levels Stage 3 introduced, now a row at a time:

```text
X @ W + b              ->  logits         any real numbers   ten raw scores
softmax(logits)        ->  probabilities  positive, sum 1    ten confidences
argmax(probabilities)  ->  prediction     one integer        the decision
```

Most examples below use three classes rather than ten, because three numbers can
be checked in your head.  Nothing in the code depends on the class count.

### Part A: why not just ten sigmoids

You already own a function that turns a score into a probability, so the obvious
move is to run ten of them:

```python
probabilities = sigmoid(logits)      # (batch, 10), Stage 3's sigmoid, unchanged
```

Do that on one row of ten logits and look hard at the result.  You get ten
numbers in `(0, 1)`, and nothing forces them to sum to anything in particular.
The model is free to say "90% sure it is a 3" and "90% sure it is an 8" about
the same image.

The problem is not that the numbers look untidy.  It is that the classes never
*compete*.  Evidence for `3` ought to be evidence against `8`, and ten
independent sigmoids offer no channel through which one class can take
probability away from another.

Both designs exist and both are used:

```text
multi-label:  several answers can be true at once  ten sigmoids, ten losses
multi-class:  exactly one answer is true           one softmax over ten logits
```

"Does this photo contain a cat?  A dog?  A tree?" is multi-label — the answers
are independent.  "Which digit is this?" is multi-class: the image has exactly
one right answer, so the model should emit a **distribution** over the ten
possibilities, not ten unrelated opinions.

So write down what the output layer has to do.  Given ten arbitrary real
numbers, produce ten numbers that are

1. positive — they are probabilities;
2. summing to one — exactly one class is correct;
3. order preserving — the largest logit gets the largest probability;
4. smooth — Stages 1 through 3 all needed derivatives, and so will this.

Before reading on, try two candidates and find where each one breaks:

```text
z / sum(z)
z**2 / sum(z**2)
```

The first fails on negative logits: it can return a negative "probability", and
its denominator can be zero.  The second repairs positivity but destroys sign —
logits `-5` and `+5` map to the same probability, so the model loses its ability
to say which classes it *rejects*.  What you need is a function that is positive
for every input and strictly increasing for every input.  `exp` is the simplest
such function, and it is the one that makes the derivatives in Part G collapse.

### Part B: softmax, and what the naive version does

For one row of logits `z` with `C` classes:

```text
softmax(z)[c] = exp(z[c]) / sum_j exp(z[j])
```

Read it as two steps with two different jobs.  The `exp` makes every score
positive while preserving order.  The division makes the row sum to one — and
that division is the only thing that couples the classes.  Raising one logit
raises the shared denominator, which lowers every other class.  That is the
competition Part A was missing, and it lives entirely in the normalization.

Now implement that formula literally, with no protections, and run it on two
rows:

```python
np.array([[1001.0, 1000.0,  999.0]])
np.array([[-1001.0, -1000.0, -999.0]])
```

Record exactly what you get and which warnings NumPy prints.  Both rows return
`nan`, for opposite reasons:

```text
[1001, 1000, 999]     exp -> [inf, inf, inf]   sum -> inf   inf/inf -> nan
[-1001, -1000, -999]  exp -> [0.,  0.,  0.]    sum -> 0.    0./0.   -> nan
```

Then compute what the answers *should* have been.  Both rows are ordinary rows
in disguise: the first has the same class differences as `[2, 1, 0]` and the
second the same as `[0, 1, 2]`, so the correct probabilities are
`[0.665, 0.245, 0.090]` and `[0.090, 0.245, 0.665]`.  Nothing about these inputs
is mathematically hard.  Only the implementation failed.

Two things to notice, because they change how much you should care:

- Compare this with Stage 3.  There the naive sigmoid returned the *right*
  answers at `+/-1000` and printed only a warning, so an `isfinite` test could
  not tell a naive implementation from a careful one.  Here the naive softmax
  returns `nan`, and `nan` is not a warning you can shrug off.  It spreads:
  `mean` of anything containing `nan` is `nan`, `nan < 1.0` is `False`, and even
  `nan == nan` is `False`, so a "keep the best loss so far" check silently stops
  updating.  Once a `nan` reaches a parameter it stays there for the rest of
  training.
- By default NumPy warns about the overflow and about `inf/inf` ("invalid value
  encountered in divide") but says nothing at all about the underflow —
  underflow is ignored unless you ask for it with `np.seterr(under="warn")`.
  Turn it on and rerun the second row.  Silent underflow will come back in Part
  D.

One more objection to answer, because it is the right objection: is a logit of
`1000` realistic?  A logit is a dot product of 784 pixel values with 784
weights.  With raw pixels in `0..255` and no care taken over the weight scale, a
few hundred is easy — which is why Stage 7 divides the images by 255.  The more
common trigger is a learning rate that is slightly too large: the weights grow
over a handful of steps, and the first symptom you ever see is a loss of `nan`.

### Part C: the shift that makes softmax safe

Subtract the largest logit in each row before exponentiating:

```text
shifted = z - max(z)          per row
```

First convince yourself this changes nothing.  For any `k` that is constant
*within a row*:

```text
exp(z[c] + k)              exp(k) * exp(z[c])              exp(z[c])
--------------------  =  -------------------------  =  ----------------
sum_j exp(z[j] + k)        exp(k) * sum_j exp(z[j])        sum_j exp(z[j])
```

The common factor `exp(k)` cancels out of numerator and denominator.  This is an
identity, not an approximation: the shifted computation returns the *same*
answer, so nothing is being traded away for stability.

Verify it in code, and use a different constant for each row so that the test is
not accidentally weaker than it looks:

```python
logits = np.array([[2.0, -1.0, 0.5], [4.0, 1.0, -3.0]])
shifts = np.array([[1000.0], [-800.0]])          # note the (batch, 1) shape
softmax(logits) == softmax(logits + shifts)
```

Three consequences worth saying out loud:

- **Only differences matter.**  Softmax depends on the logits only through their
  within-row differences.  A row of ten logits therefore carries just nine
  independent numbers; the tenth direction — adding the same amount to all ten —
  is invisible to the probabilities and so to the loss.  Remember this, because
  "every row of the gradient sums to zero" in Part G is the same fact viewed
  from the derivative side.
- **Overflow becomes impossible, not merely unlikely.**  After subtracting the
  row maximum the largest shifted logit is exactly `0`, so the largest
  exponential is exactly `1`.  The denominator is then at least `1` (the
  maximum's own term) and at most `C` (all classes tied).  A bounded denominator
  is a guarantee, not a hope.
- **Underflow survives, and here it is harmless.**  Entries more than about 746
  below the row maximum still round to `0.0`, but a class that far behind
  honestly deserves probability zero.  Part D is where the same rounding stops
  being harmless.

Then answer this: subtracting the maximum is a *choice*, since any per-row
constant cancels.  Would subtracting the row *mean* also be safe?  Test it on
`[[1e6, 0.0, 0.0]]` and explain the result.  The maximum is the only shift that
bounds every exponent above by zero for every possible input.

### Part D: log_softmax, and why not log(softmax(z))

Stage 3 taught this lesson at the loss: build probabilities first and you
destroy information the logits still hold.  Here is the same lesson one level
up, and this time you can see the loss happen in a single print.

Run:

```python
logits = np.array([[0.0, -1000.0]])
softmax(logits)             # -> [[1.0, 0.0]]
np.log(softmax(logits))     # -> [[0.0, -inf]]
```

The true answer for the second entry is about `-1000`, and `-1000` is a
perfectly ordinary float64.  Nothing was too large or too small to represent.
The number was destroyed by the round trip: `exp` squeezed it into `0.0`, and
`log` could not bring it back.  Anything more than about 746 below the row
maximum suffers this.

So compute the logarithm directly rather than as the log of a quotient:

```text
log_softmax(z)[c] = z[c] - max(z) - log( sum_j exp(z[j] - max(z)) )
```

Every term here is safe: the exponentials are all at most `1` by Part C, and the
sum is at least `1`, so the logarithm is applied to a number in `[1, C]`.  The
quantity `max(z) + log(sum_j exp(z[j] - max(z)))` is called **log-sum-exp**; you
have already used its two-term special case, `np.logaddexp`, in Stage 3.

Check this identity by hand before implementing it — it is the whole idea in one
line:

```text
log softmax(z)[c] = z[c] - logsumexp(z)
```

Implement:

```python
def log_softmax(logits: np.ndarray) -> np.ndarray:
    ...


def softmax(logits: np.ndarray) -> np.ndarray:
    ...
```

Define `softmax` as `np.exp(log_softmax(logits))`.  One stable primitive,
written once and used for both — the same structure as Stage 3, where
`np.logaddexp` gave you both the sigmoid and the loss.  Then confirm on the two
rows from Part B that your `softmax` returns `[0.665, 0.245, 0.090]` and
`[0.090, 0.245, 0.665]` with no `nan` anywhere.

Be clear about which direction is lossy, because only one of them is.  Going
`log_softmax -> exp -> softmax` rounds a very negative log-probability to `0.0`,
which is the correct probability to five hundred decimal places.  Going
`softmax -> log` rounds it to `-inf`, which is not the correct log-probability
by any margin at all.  Part F needs log-probabilities, so it must never take
that second route.

Finally, state what an `-inf` costs you downstream: the mean of a batch
containing one `-inf` is `inf`, the gradient of an infinite loss is `nan`, and a
`nan` parameter produces `nan` logits forever after.  A wrong number can be
trained away.  A `nan` cannot.

### Part E: axis and keepdims

This part is about NumPy rather than machine learning, and it is where most of
the real debugging time in Stage 4 goes.  Both reductions in `log_softmax` — the
maximum and the sum — are reductions *over classes*, which means `axis=1`:

```text
axis=0   runs down the batch      -> one number per class    (never right here)
axis=1   runs across the classes  -> one number per example  (what softmax uses)
```

Now the trap.  Compare the two shapes:

```python
np.max(logits, axis=1)                  # (batch,)
np.max(logits, axis=1, keepdims=True)   # (batch, 1)
```

Broadcasting lines shapes up starting from the *right-hand* end.  So
`(batch, classes) - (batch,)` tries to match the batch axis against the class
axis, which is never what you meant.  What happens next depends on a number that
has nothing to do with your intent:

- `(2, 3) - (2,)` raises
  `operands could not be broadcast together with shapes (2,3) (2,)`.  Loud,
  immediate, fixed in a minute.
- `(10, 10) - (10,)` is a legal broadcast, so NumPy silently subtracts the wrong
  thing.  A batch of ten MNIST images has `batch == classes == 10`.

Run the silent case and read the numbers:

```python
logits = np.arange(9.0).reshape(3, 3)
np.max(logits, axis=1)              # [2., 5., 8.]
logits - np.max(logits, axis=1)     # down the columns, not across the rows
```

Finish the softmax both ways on that input.  The correct version gives every row
`[0.090, 0.245, 0.665]`; the version without `keepdims` gives every row
`[0.867, 0.117, 0.016]` — the ranking is *reversed*, so `argmax` returns the
wrong class for every example.

Then draw the conclusion about testing, which matters more than the bug itself.
The buggy rows still sum to one, because the normalization happens after the
damage, so `rows sum to one` passes.  A test that cannot fail is not evidence.
Two tests do catch it: comparing against a per-row Python loop, and checking
that `argmax(logits) == argmax(softmax(logits))`.  Write at least one of them.

### Part F: cross-entropy over ten classes

Nothing new has to be invented here.  Stage 3 built the definition already:

```text
H(p, q) = -sum_x p(x) log q(x)     p picks the outcomes, q sets the price
```

Only the range of the sum changes.  `q` is now your row of ten probabilities,
and `p` is the label written as a distribution that puts all its mass on the
true class:

```text
label 3  ->  p = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
```

Substitute that into the definition.  Nine of the ten terms are multiplied by
zero, so the sum collapses to exactly one surviving term:

```text
loss for one example = -log( probability the model gave the correct class )
```

That is the same sentence you learned in Stage 3, word for word.  Ten classes
changed the arithmetic and left the idea alone.

**One-hot labels versus integer labels.**  You could build the `p` vector
literally and take a dot product:

```python
one_hot = np.eye(10)[labels]                       # (batch, 10)
loss = -np.mean(np.sum(one_hot * log_probabilities, axis=1))
```

Do that once and check it against the version below, because seeing them agree
is what connects the definition to the code.  Then stop doing it.  It builds a
`(batch, 10)` array to hold `batch` useful numbers, and nine of every ten
multiplies are `0 * something`.  Since you already know which entry survives, go
get it:

```python
log_probabilities[np.arange(batch_size), labels]
```

**Read that indexing carefully**, because it is the line students most often
write from memory and least often understand.  Two integer arrays used as
indices are *zipped*, not crossed.  With `labels = np.array([0, 2])` it returns
`[log_probabilities[0, 0], log_probabilities[1, 2]]` — element `i` of the first
array pairs with element `i` of the second.  Print the shapes of all three of
these on a `(2, 3)` array and explain each one:

```python
log_probabilities[np.arange(2), labels]   # (2,)   one per example  <- correct
log_probabilities[:, labels]              # (2, 2) every row, columns 0 and 2
log_probabilities[labels]                 # (2, 3) rows 0 and 2, whole rows
```

The middle one is the classic bug.  It runs, it returns floats, and its shape is
`(batch, batch)` — which is indistinguishable from `(batch, classes)` on exactly
the batch size you are most likely to debug with, ten.

Now implement:

```python
def cross_entropy_from_logits(
    logits: np.ndarray,
    labels: np.ndarray,
) -> float:
    ...
```

Take `log_softmax` of the logits, gather the true class from each row, and
return the negative mean.  Note what the function does *not* do: it never calls
`softmax`, and it never calls `np.log`.  Part D explains why that matters.

**Learn the landmark numbers.**  Because the loss is `-log(p_true)`, you can
read it back as a probability.  Memorize the middle rows:

```text
p_true    loss (nats)
0.01      4.605
0.10      2.303      <- a ten-class model that has learned nothing
0.50      0.693      <- a two-class model that has learned nothing   (Stage 3)
0.90      0.105
0.99      0.010
```

`log(10) = 2.3026` is to Stage 4 what `log(2) = 0.6931` was to Stage 3.  When
you train MNIST in Stage 8, the loss at step 0 should be very close to `2.303`;
if it is `8.0` instead, your model is confidently wrong before it has seen any
data, which means the initialization is broken rather than the training.  This
one number is the cheapest diagnostic in the course.

The inverse reading is just as useful: `exp(-loss)` is the geometric mean of the
probability your model assigned to the correct answer.  A loss of `0.10` means
"on average I gave the right digit about 90% of my belief".

Finally, be deliberate about the `mean`.  Averaging over the batch rather than
summing keeps the loss — and the gradient — the same size whether the batch
holds 32 examples or 512, so a learning rate tuned on one batch size still works
on another.  With `sum` you would have to shrink the learning rate in proportion
to the batch size to get the same steps.  This is where the `/ batch_size` in
Part G comes from.

### Part G: the gradient at the logits

Implement one fused function that returns everything at once:

```python
def softmax_cross_entropy_with_gradient(
    logits: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    ...
```

The gradient is:

```text
dloss/dlogits = ( probabilities - one_hot(labels) ) / batch_size
```

Compare that with `dloss/dlogit = p - y` from Stage 3, and with the gradients
from Stages 1 and 2.  The model changed, the loss changed, the number of classes
changed, and the shape of the answer did not.  Softmax and cross-entropy are a
matched pair in exactly the way sigmoid and binary cross-entropy were.

**Predict the properties before you derive anything.**  Each of these should be
checkable without algebra:

- Every row sums to zero, since `sum(probabilities) = 1` and `sum(one_hot) = 1`.
  This is Part C's invariance seen from the other side: the loss does not change
  when you add a constant to a whole row, so it has zero slope in that
  direction, so the gradient has no component along it.
- The true class entry is `p_true - 1`, always in `(-1, 0)`, so it is negative.
  Gradient descent subtracts the gradient, so the true class logit goes **up**.
- Every wrong class entry is `p_c`, always positive, so those logits go
  **down**.  The row redistributes belief; it does not create it.
- Sizes are self-limiting in the right direction.  A confidently *wrong* example
  gives about `-1` on the true class and about `+1` on the class it picked — the
  largest correction the formula can produce.  A confidently *right* example
  gives a row of nearly zeros and moves almost nothing.  This is the property
  Stage 3 showed squared error destroys.
- The gradient array has the same shape as `logits`.  It always does.  Stage 3
  called this the most useful debugging check in the course, and that has not
  changed.

**Then derive it**, because it is only three lines.  Write `y` for the true
class of one example, `L = -log p_y`, and use the softmax Jacobian:

```text
dp_c / dz_j = p_c * (delta_cj - p_j)          delta_cj is 1 if c == j, else 0
```

Apply the chain rule to `L = -log p_y`:

```text
dL/dz_j = -(1 / p_y) * dp_y/dz_j
        = -(1 / p_y) * p_y * (delta_yj - p_j)
        = p_j - delta_yj
```

The `1 / p_y` from the logarithm cancels the `p_y` from the softmax exactly,
which is the whole reason this pair is used together — and the reason no `1 / p`
ever appears in your code, where it would explode for a confidently wrong
prediction.  Stack that over the batch and divide by `batch_size` for the mean,
and you have `(probabilities - one_hot(labels)) / batch_size`.  The Jacobian
step itself is the stretch exercise; do the two-case check `c == j` and `c != j`
on paper once.

**Now write it, and watch out for the aliasing.**

```python
gradient = probabilities.copy()
gradient[np.arange(batch_size), labels] -= 1.0
gradient /= batch_size
```

The `.copy()` is not defensive style, it is required.  In Python
`gradient = probabilities` binds a second name to the same array — it never
copies — so the in-place `-=` on the next line would subtract `1.0` from the
probabilities you are about to return.  The caller then sees a "probability" of
`-0.83` and has no way to tell where it came from.  The same hazard applies to
slices, which are views rather than copies:

```python
row = probabilities[0]      # a view: row -= 1 modifies probabilities
row = probabilities[0].copy()
```

Return the loss, the probabilities, and the gradient together from this one
function.  Softmax then gets computed once instead of twice, the loss goes
through `log_softmax` and never through `log(softmax(...))`, and the
cancellation above means the fused form never divides by a probability.  Every
real framework ships this exact fusion — `softmax_cross_entropy_with_logits` in
TensorFlow, `CrossEntropyLoss` in PyTorch, which is why both take logits and not
probabilities.

**Check it numerically.**  There are no parameters yet, so differentiate with
respect to the logits themselves.  For each entry `(i, j)`, hold everything else
fixed and compare:

```text
                        loss(z with z[i,j]+eps) - loss(z with z[i,j]-eps)
dloss/dlogits[i,j]  ~=  ----------------------------------------------------
                                            2 * eps
```

With `eps = 1e-6` and float64 the agreement should be around `1e-10`.  This is
the same central-difference check you wrote in Stages 1 through 3, and it is the
only part of Stage 4 that proves your gradient is the derivative of your loss
and not merely a plausible formula.

### Part H: predicting, and what softmax is actually for

Implement:

```python
def predict(logits: np.ndarray) -> np.ndarray:
    ...
```

Return `np.argmax(logits, axis=1)` — one integer per row.

Notice what is missing: the prediction never calls softmax.  Since `exp` is
strictly increasing and every row is divided by the same denominator,
`argmax(logits) == argmax(softmax(logits))` for every row.  Verify that on
random logits.  So at prediction time softmax is pure overhead, and a deployed
classifier that only reports a class can skip it entirely.

Then answer the obvious follow-up: if you never need softmax to classify, why
does it exist?  Two reasons, and both are Stage 3's answers again.  `argmax` is
a step function of the logits — constant over every small change, so its
derivative is zero and carries no signal, and discontinuous where the winner
changes.  It cannot train anything.  Softmax is the differentiable stand-in.
And a probability answers "how sure?", which a class index cannot.

Finally, see what *scaling* the logits does, since Part C only made you check
adding to them:

```python
base = np.array([[2.0, 1.0, 0.1]])
softmax(base / 10.0)   # [0.366, 0.331, 0.303]   nearly uniform
softmax(base / 1.0)    # [0.659, 0.242, 0.099]
softmax(base / 0.1)    # [1.000, 0.000, 0.000]   nearly one-hot
```

The prediction is identical in all three.  Only the confidence changes.  That
divisor is called the *temperature*, and this experiment explains the name
"softmax": at low temperature it approaches a hard `argmax` — a one-hot vector —
and at high temperature it melts toward the uniform distribution.  It is a soft,
differentiable version of "pick the biggest", which is precisely what Part A
asked for.

Keep the consequence in mind for Stage 11: a network can grow more and more
confident, driving the loss down, long after its accuracy has stopped moving.
You saw the same split in Stage 3.

### Minimum tests

A test that passes for a broken implementation has taught you nothing, so a few
of these are written specifically to fail against the mistakes in Parts B, D and
E.

- Every softmax row sums to one.
- All probabilities are finite for logits of `1000`, where the naive formula
  returns `nan`, and they equal the answer for the same row shifted down to
  ordinary values.
- Adding a *different* constant to each row does not change the probabilities.
- On a square batch — say `(10, 10)` — your softmax matches a plain Python loop
  that computes one row at a time.  This is the `keepdims` test; the rows will
  sum to one either way.  Give the input distinct row maxima, or the test is
  vacuous: if every row peaks at the same value, subtracting down the columns
  happens to agree with subtracting across the rows and the bug hides.
- `argmax(logits) == argmax(softmax(logits))` on random logits.
- `log_softmax` is finite at `[[0.0, -1000.0]]`, where `np.log(softmax(...))`
  returns `-inf`.
- Two-column softmax agrees with Stage 3's sigmoid of the logit difference.
- Gathering with `np.arange` indexing equals the one-hot dot product.
- Uniform logits over ten classes give a loss of `log(10) = 2.3026`.
- Increasing the true class logit decreases the loss.
- Analytic `dL/dlogits` agrees with central differences to about `1e-10`.
- `dL/dlogits` has the same shape as `logits`.
- Each row of `dL/dlogits` sums to zero.
- The returned probabilities are untouched by the gradient computation — they
  still sum to one and are still non-negative after the function returns.
- Invalid labels and shapes produce useful errors: a label of `10` with ten
  classes, a negative label, one-dimensional logits, and a label count that does
  not match the batch size.

### Questions to answer

1. Why does softmax use all ten logits rather than transforming each
   independently?
2. Why are logits often more convenient than probabilities inside a model?
3. What is the cross-entropy loss when all ten probabilities are exactly `0.1`?
   What loss do you therefore expect at step 0 of MNIST training, and what would
   a step-0 loss of `8.0` tell you?
4. Why does the true class's gradient entry usually have a negative sign?
5. What does a positive gradient for a wrong class tell gradient descent to do?
6. Softmax depends only on differences of logits.  How many independent numbers
   does a row of ten logits really carry, and how does that fact show up in the
   gradient?
7. Subtracting the row maximum changes nothing, but subtracting a per-*column*
   maximum changes everything.  Why is one free and the other a bug?
8. Write Stage 3 as the two-class case of this stage: show that
   `softmax([z0, z1])[1] = sigmoid(z1 - z0)`.  Which quantity plays the role of
   the single Stage 3 logit?
9. Your loss is `0.10` nats.  What average probability is the model giving to
   the correct digit?
10. Subtracting the maximum removes overflow but leaves underflow.  Why is the
    surviving underflow harmless inside `softmax` and fatal inside
    `log(softmax(...))`?
11. Where does the `/ batch_size` in the gradient come from, and what would you
    have to change about the learning rate if the loss used `sum` instead of
    `mean`?

### Stretch

Derive the `probabilities - one_hot(labels)` result from the softmax derivative
and cross-entropy derivative.  The implementation may use the simplified result,
but work through the cancellation once on paper.  Start from
`p_c = exp(z_c) / sum_j exp(z_j)` and do the two cases `c == j` and `c != j`
separately; the quotient rule gives both, and they combine into the single
Jacobian in Part G.

Replace the one-hot label with a *smoothed* one, say `0.9` on the true class and
`0.1 / 9` spread over the rest, and compute the loss from the full definition
`-sum_c p_c log q_c`.  Now every term survives instead of just one.  Show that
the gradient becomes `probabilities - p` for that softer `p`, and say what the
model's best possible prediction is now — this is the answer to Stage 3's
question about what would have to be true of the labels for the other terms to
survive.

Compute the entropy of the model's own predicted distribution,
`-sum_c p_c log p_c`, for a confident row and an unsure row.  You now have a
number that measures how uncertain the model is *without knowing the label*,
which is the basis of most "flag this one for a human" systems.

Time the one-hot matmul form against the `np.arange` gather on a `(60000, 10)`
array and report the ratio.  Then say which of the two costs you would actually
notice on MNIST, and which one you would notice with a 50,000-word vocabulary
instead of ten digits.

