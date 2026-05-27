## Stage 3 — A single neuron becomes a binary classifier

### Goal

Move from predicting a number to predicting a category.  Introduce logits,
probabilities, sigmoid, binary cross-entropy, and a decision rule.

Notice first what does *not* change.  The training loop is still:

```text
compute predictions
compute loss
compute gradients
parameter -= learning_rate * gradient
```

Only the model output and the loss function change.  That stays true for every
remaining stage, including the final MNIST network.

### Keep three levels separate

Most Stage 3 confusion is a mix-up between these:

```text
X @ weights + bias   ->  logit         any real number       raw score
sigmoid(logit)       ->  probability   strictly in (0, 1)    confidence
probability >= 0.5   ->  prediction    0 or 1                the decision
```

Stage 2 stopped at the first line and called it the answer.  Classification keeps
going.  Note where each function below operates: the loss and the gradient both
live at the **logit** level and never touch probabilities.  Part F explains why.

Generate two labeled clusters in two dimensions:

```text
class 0 centered near (-2, -2)
class 1 centered near ( 2,  2)
```

Each example has two features:

```text
X:      (batch, 2)
labels: (batch,)      values are 0 or 1
```

### Part A: logits

Use parameters:

```text
weights: (2,)
bias:    scalar
```

Compute:

```text
logits = X @ weights + bias
```

This is exactly the Stage 2 model with two features instead of one.  A logit is
an unrestricted score.  It is not yet a probability.

### Part B: sigmoid

Implement:

```python
def sigmoid(logits: np.ndarray) -> np.ndarray:
    ...
```

Sigmoid maps every real number into `(0, 1)` and maps `0` to `0.5`:

```text
sigmoid(z) = 1 / (1 + exp(-z))
```

First implement that directly and evaluate it on:

```python
np.array([-1000.0, 0.0, 1000.0])
```

Record two separate things: the values you get back, and any warning NumPy
prints.  The values are correct.  Only the warning is a problem, and you should
be able to say exactly which intermediate quantity overflowed to produce it.

Now verify this identity by hand,

```text
1 / (1 + exp(-z))  ==  exp( -log(1 + exp(-z)) )
```

and implement it with `np.logaddexp(0.0, -logits)`, which computes
`log(1 + exp(-z))` without ever forming the exponential.  One line is enough; no
branching or masking is needed.

The direct and stable versions must agree to floating-point precision on ordinary
inputs such as `[-500, -8, 0, 8, 500]`.  The stable version must additionally
produce no overflow warning at `+/-1000`.

### Part C: where the name "cross-entropy" comes from

The formula in Part F looks arbitrary until you build it from one idea, so build
it.  Everything below is about *measuring a probabilistic prediction*; none of it
is specific to neural networks.

Start with a single outcome and how surprised you are to see it.  Your model
assigned it probability `q`.  Define:

```text
surprise(outcome) = -log q(outcome)
```

Check that it behaves the way surprise should.  An outcome you called certain
(`q = 1`) has surprise `0` — you learned nothing.  An outcome you called
impossible (`q -> 0`) has surprise `-> infinity`.  And for two independent
events, probabilities multiply while surprise adds:

```text
-log(q1 * q2) = -log q1 + -log q2
```

Those three requirements are not merely satisfied by `-log q`; they pin it down.
Any function that is zero at `q = 1` and turns products into sums must be a
multiple of the logarithm, and the multiple is only a choice of unit.  Use
`log2` and the unit is *bits*; use natural log and the unit is *nats*.  This
course uses nats throughout, so remember that `log(2) = 0.6931` nats is exactly
one bit.

Now average the surprise over many outcomes.  Two distributions can appear, and
keeping them apart is the whole point:

```text
p   the distribution outcomes are really drawn from   (reality)
q   the distribution your model predicts              (your model)
```

If your model happens to be exactly right, so that `q = p`, the average surprise
is

```text
H(p) = -sum_x p(x) log p(x)              <- entropy
```

That is Shannon's entropy: the irreducible average surprise of the source.  You
cannot do better, no matter how good your model is, because the randomness is in
the world and not in you.

If your model is *wrong*, reality still decides which outcomes occur and how
often — that is `p` — but you get charged `-log q(x)` for each one, because `q`
is what you predicted.  The average is

```text
H(p, q) = -sum_x p(x) log q(x)           <- cross-entropy
```

Two different distributions in one expression, each playing a different role:
`p` picks the outcomes, `q` sets the price.  That crossing is the name.  Read it
out loud as "the average surprise of reality, scored by my model."

Now the fact that makes it a loss function.  Cross-entropy is never smaller than
entropy, and equals it only when `q = p`.  The gap has a name:

```text
H(p, q) = H(p) + KL(p || q)
          ^^^^   ^^^^^^^^^^
          fixed  your error, >= 0, zero only if q == p
```

`H(p)` does not depend on your parameters at all, so minimizing cross-entropy is
*exactly* minimizing `KL(p || q)` — pushing your predicted distribution onto the
true one.  That is the justification, in one line.  You are not minimizing an
arbitrary distance; you are minimizing the excess surprise your model causes.

### Part D: from that definition down to our two-term formula

Apply the definition to supervised learning.  We never see the true `p`; we see a
label.  Treat the label as a distribution that puts all its mass on the observed
class:

```text
label y = 1  ->  p = [0, 1]        p(class 0) = 0,     p(class 1) = 1
label y = 0  ->  p = [1, 0]
in general    p = [1 - y, y]
```

The model's distribution is `q = [1 - p_hat, p_hat]` where `p_hat = sigmoid(z)`.
Substitute both into `H(p, q) = -sum_x p(x) log q(x)`:

```text
H(p, q) = -(1 - y) * log(1 - p_hat) - y * log(p_hat)
```

That is Part F's formula, derived rather than asserted.  Confirm two things
before moving on:

- The sum over classes collapsed to a single surviving term, because `p` is zero
  on every class but the true one.  With hard labels, cross-entropy always
  reduces to `-log(probability the model gave the correct answer)`.  Say that
  sentence back without looking; it is the whole loss.
- `H(p) = 0` for a one-hot `p` — a certain outcome carries no surprise.  So here
  the cross-entropy *is* the KL divergence, with nothing else mixed in.  The
  number you print during training is pure model error.

Now forget information theory entirely and ask a statistician's question: which
parameters make the labels we actually observed most probable?  The likelihood of
the whole dataset is

```text
prod_i  p_hat_i**y_i  *  (1 - p_hat_i)**(1 - y_i)
```

Take the negative log, divide by `N`, and you have the mean binary cross-entropy
again.  Minimizing cross-entropy *is* maximum likelihood estimation.  Information
theory and statistics arrive at one formula by different routes, which is the
strongest evidence you will get that the formula is not a design choice.

One notation warning before you continue.  In the two sections above, `p` is
*reality* and `q` is *your model* — that is the standard convention in
information theory, and every textbook you read will use it.  In the rest of this
stage, and in almost all neural-network code, `p` instead means the model's own
probability `sigmoid(z)`, and the label is `y`.  The letters are reused and the
meanings do not carry over.  So when you read `dloss/dlogit = p - y` below, that
`p` is the `p_hat` from this section.

Because the units are real, the loss is readable.  Learn these three landmarks:

```text
0.6931 nats = log(2)    one bit    a two-class model guessing 0.5      (Stage 3)
2.3026 nats = log(10)              a ten-class model guessing 0.1      (Stage 4)
0.05   nats                        a model that is almost never surprised
```

You will meet `0.6931` again as the step-0 loss in Part H.  When you do, you
should already know why: zero parameters give logit `0`, sigmoid gives `0.5`, and
one bit of uncertainty per example is precisely a coin flip.  A squared error of
`0.25` tells you nothing comparable.

### Part E: why cross-entropy and not squared error

You already have a loss that works on numbers: `0.5 * (p - y)**2`.  Everything
above argued that cross-entropy is *meaningful*; it did not yet argue that
squared error is unusable.  In fact squared error on probabilities is a perfectly
honest score — like cross-entropy, it is minimized only when the predicted
probability equals the true one, so it will not reward a dishonest model.  The
problem is not what it measures.  It is what it does to *learning*.

For a sigmoid neuron the derivative of the loss with respect to the logit is:

```text
cross-entropy:  dloss/dlogit = p - y
squared error:  dloss/dlogit = (p - y) * p * (1 - p)
```

Evaluate both for an example the model gets confidently wrong — label `1` at
logit `-20` — and compare the magnitudes.  Then answer in one sentence: what does
squared error do when the model is maximally wrong, and why is that the opposite
of what you want?

The extra `p * (1 - p)` is the sigmoid's own derivative, and it goes to zero at
both ends.  Squared error keeps it; cross-entropy's `log` produces exactly its
reciprocal and cancels it, which is what Part G shows.  So the choice of loss is
what decides whether a saturated neuron can still learn.

This is the reason classification uses cross-entropy.  It is not a convention.

### Part F: a stable loss, computed from logits

Binary cross-entropy for one example is:

```text
-y * log(p) - (1 - y) * log(1 - p)
```

Substitute `p = sigmoid(z)` and simplify by hand.  You should reach:

```text
log(1 + exp(z)) - y * z
```

Implement the mean of that:

```python
def binary_cross_entropy_from_logits(
    logits: np.ndarray,
    labels: np.ndarray,
) -> float:
    ...
```

NumPy provides the stable first term as `np.logaddexp(0.0, logits)`.

The two forms are algebraically identical, so it is fair to ask why the second is
worth the trouble.  Evaluate both at:

```python
logits = np.array([-800.0, 800.0])
labels = np.array([1.0, 0.0])
```

The first form builds probabilities, which round to exactly `0.0` and `1.0`, and
then takes `log(0.0)`.  The second returns a finite answer.  A perfectly stable
sigmoid does not rescue the first form, because `sigmoid(-800)` really is `0.0`
in float64 — the information is destroyed the moment you leave logit space.
This, and not Part B, is where numerical stability decides whether training works
at all.

Then find the smallest positive integer logit at which `1.0 - sigmoid(z)` becomes
exactly zero, and the most negative integer logit at which `sigmoid(z)` becomes
exactly zero.  The two answers are nowhere near each other, and the smaller one
is a value real training reaches easily.  Explain the asymmetry: why does float64
run out of room to represent "just below one" so much sooner than "just above
zero"?

Then compare the loss for these four cases and describe the pattern:

```text
label 1, logit  8
label 0, logit -8
label 1, logit -8
label 0, logit  8
```

Note also how the loss for a confidently wrong prediction grows as the logit
grows.  Is that growth linear or exponential?  Why does the answer matter for
keeping training stable?

### Part G: backward propagation

Sigmoid's derivative is `p * (1 - p)`, and the log in cross-entropy contributes
`1 / (p * (1 - p))`.  Multiply them and everything cancels:

```text
dloss/dlogits = (probabilities - labels) / batch_size
```

Compare this with the gradients from Stages 1 and 2.  The model changed and the
loss changed, and the form did not.  Sigmoid and cross-entropy are a matched
pair, chosen for each other.

Use the affine-layer derivatives:

```text
dloss/dweights = X.T @ dloss/dlogits
dloss/dbias    = sum(dloss/dlogits)
```

Check every shape before running the calculation:

```text
dloss/dlogits    (batch,)      one number per example
X.T              (2, batch)
X.T @ dlogits    (2,)          same shape as weights
sum(dlogits)     scalar        same shape as bias
```

A gradient always has the same shape as the parameter it belongs to.  That is the
most useful debugging check in this course.

Finally, write `X.T @ dloss/dlogits` out as an explicit sum over examples and
confirm it is the Stage 2 gradient generalized to more than one feature.

### Part H: train and classify

Start weights and bias at zero.  Repeat full-batch gradient descent.  Predict:

```text
class 1 if probability >= 0.5
class 0 otherwise
```

Track both loss and accuracy.  Three checkpoints:

- The loss at step 0 should be `0.6931`.  Predict that number before running the
  code and explain where it comes from.
- Accuracy should reach 100% within a few steps and then stop changing, while the
  loss keeps falling for hundreds more steps.  Record both curves.
- The learned `weights` should point from the class-0 cluster toward the class-1
  cluster, and `bias` should be near zero.  Explain both from the geometry of the
  data.

The loss will never reach zero.  Explain what the model is still improving.

### Minimum tests

- The direct and stable sigmoids agree on ordinary inputs such as
  `[-500, -8, 0, 8, 500]`.
- The stable sigmoid raises no overflow warning at `+/-1000`.
- Binary cross-entropy built from probabilities is not finite at logits `+/-800`,
  while the from-logits version is.
- Probabilities lie between zero and one.
- Correct confident logits have much lower loss than incorrect confident logits.
- Analytic gradients agree with central differences for both the weights and the
  bias.
- Gradient arrays have the same shapes as their parameters.
- The initial loss with zero parameters is `log(2)`.
- Training reaches high accuracy on the synthetic clusters.

### Questions to answer

1. Why not train directly on the thresholded class prediction?
2. What information does a probability contain that an accuracy bit does not?
3. Why can accuracy stay unchanged while cross-entropy improves?
4. What geometric object is defined by `X @ weights + bias = 0` in two dimensions?
5. Why is a single sigmoid neuron still a linear classifier despite the nonlinear
   sigmoid?
6. Why is the initial loss `log(2)` rather than some other number?
7. Zero initialization works here.  Predict whether it will still work once a
   hidden layer of several neurons is added in Stage 9, and say why.
8. Cross-entropy and KL divergence differ by `H(p)`.  Why does that difference
   change nothing about which parameters gradient descent finds?
9. The definition of cross-entropy sums over every class.  Why does that sum
   leave only one term standing here, and what would have to be true of the
   labels for the other term to survive?
10. In what units is your loss measured, and what would a loss of `0.10` mean
    stated in those units?

### Stretch

Plot the points and the learned decision boundary.  Color the background by
predicted probability.  Watch the coloring sharpen as training continues even
after the boundary itself has stopped moving.

Implement sigmoid a second way, branching on the sign of `z`: use
`1 / (1 + exp(-z))` when `z >= 0` and `exp(z) / (1 + exp(z))` when `z < 0`.  Show
the two branches are algebraically equal, then check the result against your
one-line version across the full range.
