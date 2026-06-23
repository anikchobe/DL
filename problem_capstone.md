# Capstone experiments

After the implementation works, choose several controlled experiments.  Change one
factor at a time and record the hypothesis before running it.  Keep a short
experiment log: hypothesis, setting, result, explanation.

Do these before the extensions.  They need no new code, and they build the
habit — one factor, a hypothesis written first, a baseline you ran yourself —
that every extension then depends on.

## Experiment 1: learning rate

Try one rate that is clearly too small, one useful rate, and one that makes loss
oscillate or diverge.  Explain the behavior in terms of gradient magnitude and
step size.

## Experiment 2: normalization

Compare pixels in `0..255` with pixels in `0..1` while keeping the learning rate
fixed.  Then ask whether changing the learning rate can compensate.  This reveals
that parameterization and optimization scale interact.

## Experiment 3: hidden width

Compare 8, 32, 64, and 128 hidden units.  Record parameter count, training time,
training accuracy, and validation accuracy.

## Experiment 4: symmetry failure

Initialize all hidden weights to zero.  Inspect two hidden columns after several
updates.  Show that they remain identical and explain why.

## Experiment 5: shuffled labels

Randomly permute training labels while preserving images.  A sufficiently large
network may eventually memorize a small shuffled set, but validation accuracy
should remain near chance.  This separates fitting capacity from learning useful
structure.

## Experiment 6: linear versus nonlinear

Compare Stage 8 and Stage 11 on the same split.  Inspect mistakes corrected by the
MLP and mistakes shared by both models.

## Experiment 7: finite precision

Compare float64 and float32 training speed and results.  Keep float64 for gradient
checking.  Explain why a small numerical difference is not necessarily a bug.

# Where to go next

`extensions/` holds seven optional problems, in this same format, for the topics
Stages 0 to 11 name but never build:

```text
1  L2 and early stopping        something to do about the overfitting you plotted
2  learning-rate schedules      decay and warmup, against a tuned constant rate
3  He, Xavier, and failure      the measurement behind sqrt(2 / fan_in)
4  calibration                  whether the model's confidence means anything
5  class imbalance              the case MNIST never forces you to handle
6  an autograd bridge           your Stage 10 backward pass, discovered at runtime
7  convolution                  the only one that changes the model itself
```

They are independent and unordered.  Experiments 1, 3 and 4 above each have a
matching extension — the learning rate leads to Extension 2, the hidden width to
Extension 1, the symmetry failure to Extension 3 — so a natural way to choose is
to follow whichever experiment produced a result you could not fully explain.

# Final oral walkthrough

Without opening the code, explain one training mini-batch from disk to updated
weights:

```text
gzip file
IDX header
uint8 image array
float32 flattened batch
first affine layer
ReLU
second affine layer
softmax and cross-entropy
four parameter gradients
four gradient-descent updates
```

For every arrow, state:

- the operation;
- the input and output shapes;
- whether the value is data, an intermediate, a parameter, a gradient, or a metric;
- whether it is needed during inference, training, or both.

