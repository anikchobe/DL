# Glossary — the words people use about machine learning

This document exists because the vocabulary of this field is much larger than the
number of ideas in it.  Most terms name something you have already built, or
something small that nobody bothered to explain when they said it.  A few name
genuinely new ideas.  This file tells you which is which.

Read it the way you would read a dictionary: by looking things up, not front to
back.  The one section worth reading straight through is Part 2, because it
answers a question the stages do not reach until the very end.

Each entry is tagged with where it lives in this course:

```text
[stage 3]      built in that stage — go reread it if the entry is thin
[stage 11]     a stage you have not reached yet
[beyond]       real, common, and not built in this course
```

A tag of `[stage 8]` when you are on Stage 4 is not a warning.  It means you can
read the entry now and understand it, and you will build the thing later.

One honest caution before the definitions.  Knowing that "recall" is
`TP / (TP + FN)` is worth almost nothing on its own; being able to say which
mistake a low recall means the model is making, and what you would change to fix
it, is worth a lot.  Vocabulary is how you talk to other people about the
mechanics.  It is not a substitute for the mechanics.

---

## Part 1 — The four levels of an answer

Stage 3 made you keep three levels apart.  Every metric in Part 2 lives at a
fourth level, one step further out, and almost all confusion about metrics is a
confusion between these levels:

```text
X @ W + b               ->  logit         any real number     the raw score
softmax(logit)          ->  probability   in (0, 1), sums 1   the confidence
argmax, or p >= 0.5     ->  prediction    a class             the decision
compare with the label  ->  metric        a summary number    the score card
```

Facts that follow from the picture, and that are worth having ready:

- The loss is computed at the **logit** level.  Stages 3 and 4 both went out of
  their way to keep probabilities out of the loss, for numerical reasons.
- A metric is computed at the **prediction** level, after a decision has been
  made.  That is why metrics are not differentiable and cannot be trained on:
  the decision threw the gradient away.  Stage 3, question 1, is this question.
- Loss and metric can move in opposite directions.  The model can grow more
  confident about answers it was already getting right — the loss falls, and not
  one decision changes, so accuracy sits still.  You saw this in Stage 3, Part H,
  and you will see it again in Stage 11's curves.
- Everything in Part 2 is computed from **predictions and labels only**.  Two
  models with wildly different confidences and identical decisions get identical
  metrics.  That is a real limitation, and calibration (Part 2) is the repair.

---

## Part 2 — Judging a classifier

### Why you have not met these words yet

MNIST has about ten percent of each digit.  When the classes are balanced like
that, accuracy is a fair summary and there is little reason to reach for anything
else — which is why this course gets to Stage 11 without needing precision or
recall.  They exist for the case MNIST does not present: when the classes are
**imbalanced**, or when the two kinds of mistake cost different amounts.

So build the case where accuracy lies.  Take your ten-class MNIST problem and ask
a Stage 3 question about it:

```text
is this image an 8?           yes / no
```

That is a binary problem with about 1,000 positives in the 10,000-image test set
and about 9,000 negatives.  Now consider the laziest possible model:

```python
def predict(images):
    return np.zeros(len(images), dtype=int)     # "no" every time
```

It never looks at a pixel.  Its accuracy is **90 percent**.

Sit with that for a moment, because it is the whole reason this vocabulary
exists.  A single number can be high while the model is worthless, and no amount
of staring at that number reveals it.  You need metrics that ask about the two
kinds of mistake separately.

### The confusion matrix, and the four counts

You will build the ten-by-ten version in Stage 11, Part E.  The two-class version
is a two-by-two table, and the four cells have names:

```text
                       model says 8        model says not-8
truly an 8          true positive  TP     false negative  FN
truly not an 8      false positive FP     true negative   TN
```

Read the names as two words that each mean something:

- the second word is **what the model said** — "positive" means it flagged the
  image as an 8;
- the first word is **whether it was right** — "false" means it was not.

So a false negative is a model that said not-8 and was wrong: a real 8 that got
away.  A false positive is a model that said 8 and was wrong: a 3 wearing a
disguise.  Those two errors are not interchangeable, and the entire point of
Part 2 is that one number cannot tell them apart.

Take one concrete result to compute with.  Suppose your classifier flags 900
images as 8s, and 800 of them really are:

```text
TP = 800      the 8s it found
FP = 100      the flagged images that were not 8s
FN = 200      the 8s it missed          (1000 real 8s - 800 found)
TN = 8900     the not-8s it left alone  (9000 - 100)
```

Every metric below is a different ratio of those same four numbers.

### Accuracy [stage 3]

```text
accuracy = (TP + TN) / (TP + FP + FN + TN)
         = (800 + 8900) / 10000
         = 0.97
```

The fraction of all decisions that were right.  It weighs both mistakes equally
and lets the majority class dominate, which is exactly how the lazy model reached
90 percent.  Note that 0.97 is barely better than 0.90 — accuracy compresses the
difference between a useless model and a decent one into seven points.

For the ten-class problem it is the honest headline number, and it is what
Stages 8 and 11 track.  On a lopsided problem, quote it next to something else or
not at all.

### Precision [beyond]

```text
precision = TP / (TP + FP)     of the images I flagged, how many were right?
          = 800 / 900
          = 0.889
```

The denominator is **everything the model said was positive**.  Precision is the
model's answer to "when you speak up, can I trust you?"  It falls when the model
cries wolf.

### Recall [beyond]

```text
recall = TP / (TP + FN)        of the real 8s, how many did I find?
       = 800 / 1000
       = 0.800
```

The denominator is **everything that truly was positive**.  Recall is the model's
answer to "how much did you miss?"  It falls when the model stays quiet.

### The one thing to actually remember

Both fractions have `TP` on top.  Only the denominator differs, and the
denominator is the whole distinction:

```text
precision:  divide by what the MODEL claimed     (its column of the table)
recall:     divide by what the TRUTH contains    (its row of the table)
```

If you remember which denominator goes with which name, you can rebuild both
formulas from scratch and never need to memorize them again.

Now go back and score the lazy model, because its numbers are instructive:

```text
TP = 0, FP = 0, FN = 1000, TN = 9000

accuracy  = 9000 / 10000 = 0.90     looks respectable
recall    = 0 / 1000     = 0.00     it found nothing
precision = 0 / 0        = nan      it never made a claim to be right about
```

Recall catches it instantly.  And notice what precision does: `0 / 0` is exactly
the `nan` you met in Stage 4, arriving here for a mundane reason — a model that
flags nothing has no claims to be judged on, so precision is genuinely
undefined, not zero.  Real metric code either reports it as undefined or defines
it as `0.0` by convention.  If you write your own, decide which and say so,
because a `nan` in a results table will otherwise get averaged into every summary
that touches it.

### The threshold is the dial between them [stage 3]

Stage 3, Part H, had you predict class 1 when `probability >= 0.5`.  That `0.5`
is a **choice**, not a law, and it is the reason precision and recall trade
against each other:

```text
threshold 0.9    flags only near-certain 8s      few FP, many FN   precision up, recall down
threshold 0.5    the default                     the balance you measured above
threshold 0.1    flags anything vaguely 8-ish    many FP, few FN   recall up, precision down
```

Two facts about the dial matter more than the definitions:

- **It is turned after training, not learned.**  The same trained parameters give
  you every point on that trade-off.  You are not choosing a better model; you
  are choosing where to sit on one model's curve, and you choose it on the
  validation set.
- **Where you sit depends on which mistake costs more.**  A spam filter that puts
  one real email in the junk folder has done more damage than one letting ten
  spams through, so it wants precision.  A screening test for a treatable disease
  would rather send ten healthy people for a second look than miss one case, so
  it wants recall.  No formula picks this for you; it comes from outside the
  model, from what the errors actually cost.

Sweeping the threshold across its whole range and plotting the result gives you
the curves people quote:

- a **precision–recall curve** [beyond], precision against recall;
- an **ROC curve** [beyond], recall against the false-positive rate
  `FP / (FP + TN)`;
- **AUC** [beyond], the area under the ROC curve.  It has a clean
  interpretation worth knowing: it is the probability that a randomly chosen
  positive gets a higher score than a randomly chosen negative.  So `0.5` is
  coin-flipping and `1.0` is a model that ranks every positive above every
  negative.

Because these sweep the threshold, they judge the model's **ranking** rather
than any one decision rule — which makes them the fair way to compare two models
before you have chosen a threshold for either.

### F1 score [beyond]

One number that refuses to let either of the other two collapse:

```text
F1 = 2 * precision * recall / (precision + recall)
   = 2 * TP / (2*TP + FP + FN)
   = 1600 / 1900
   = 0.842
```

This is the harmonic mean of precision and recall, and harmonic means are
dominated by their smaller argument.  Check that on the numbers: a model with
precision `1.0` and recall `0.01` has an arithmetic mean of `0.505` and an F1 of
`0.0198`.  The arithmetic mean would call that model half decent.  F1 correctly
calls it nearly useless, which is the entire reason the harmonic mean is the one
used here.

F1 assumes precision and recall matter equally.  When they do not, the general
form `F_beta` weights recall `beta` times as heavily; `beta = 2` favors recall,
`beta = 0.5` favors precision.  Reach for it only when you can say what the
weight means.

### Aliases you will hear

Different fields renamed the same fractions.  There is nothing new in this list:

```text
recall              = sensitivity = true positive rate = hit rate
precision           = positive predictive value
specificity         = TN / (TN + FP) = true negative rate
false positive rate = FP / (FP + TN) = 1 - specificity
miss rate           = FN / (FN + TP) = 1 - recall
```

Medicine says sensitivity and specificity; information retrieval says precision
and recall; machine learning says both depending on who trained the speaker.

### Ten classes: read them off the confusion matrix [stage 11]

Precision and recall are defined for a yes/no question, so for ten digits you get
ten of each, one per class, by treating each digit as "is it this one?"  Stage 11
builds the matrix with rows as true digits and columns as predicted digits, and
that layout makes both fractions a row sum and a column sum:

```text
recall(c)    = M[c, c] / sum of row c        of the real c's, how many did I get?
precision(c) = M[c, c] / sum of column c     of the images I called c, how many were?
accuracy     = sum of the diagonal / total
```

This is the numeric answer to Stage 11's question 4, "what does a confusion
matrix reveal that one accuracy number hides."  A model at 97 percent overall can
have 99 percent recall on 0s and 93 percent on 9s, and the per-class numbers say
where to look.  The off-diagonal cells then say what it is confusing them *with*,
which is more than a metric can: `true 9 predicted 4` and `true 5 predicted 3`
name a specific pair of digits that share strokes.

Two ways to average the per-class numbers into one, and the difference is not
cosmetic:

- **macro average** — the plain mean of the ten per-class values.  Every digit
  counts the same regardless of how many examples it has, so a rare class the
  model is bad at cannot hide.
- **micro average** — pool all ten classes' TP, FP and FN and compute once.  On a
  single-label problem like MNIST, where every image gets exactly one predicted
  class, every error is simultaneously somebody's FP and somebody's FN, and micro
  precision, micro recall, micro F1 and accuracy all collapse to the same number.
  Verify that on a small confusion matrix once; after that you will know that
  "micro-averaged F1" on this kind of problem is accuracy under a longer name.

### Two more metrics worth knowing [stage 11 stretch]

- **Top-k accuracy** — counts a prediction correct if the true class is among the
  `k` highest logits.  Top-5 accuracy is the standard on problems with a
  thousand classes, where being roughly right has value.  On ten digits, top-1 is
  what matters.
- **Calibration** — whether stated confidence matches observed frequency.  Take
  every prediction the model made with about 80 percent confidence; if it is
  well calibrated, about 80 percent of them are right.  A model can be accurate
  and badly calibrated — Stage 4, Part H showed that dividing the logits by a
  temperature changes every probability while changing no `argmax` at all, so
  confidence and correctness are genuinely separate things to measure.

### Baselines [stage 11]

Not a metric, but the thing that makes a metric mean something.  Always know the
score of a model that does nothing:

```text
uniform guessing on ten digits         10% accuracy, loss log(10) = 2.303
always guess the most common class     that class's share of the data
the lazy "never an 8" model above      90% accuracy on the is-it-an-8 problem
```

A number is not a result until you know what beating nothing looks like.  This is
why Stage 11 lists "accuracy stays near ten percent" as a symptom: ten percent is
not a low score, it is the score of no model at all.

---

## Part 3 — Words about data

**Example / sample / instance** [stage 2] — one row of your data: one image and
its label.  Three words, one meaning.  Careful: "sample" also means a whole
subset in statistics, so it is the most overloaded word in the field.

**Feature** [stage 3] — one input number describing an example.  MNIST features
are the 784 pixel intensities, and `D` in this course's shape notation is the
feature count.  In problems where features are hand-designed rather than raw,
inventing good ones is called **feature engineering** [beyond]; the reason deep
learning was a big deal is that a network learns its own features from raw input.

**Label / target / ground truth** [stage 1] — the answer you are training toward.
"Ground truth" carries a mild reminder that the labels themselves can be wrong —
MNIST has a handful of genuinely mislabeled images, which is why Stage 11 lists
"possible label surprise" as a category of mistake.

**One-hot vector** [stage 0] — a class written as a distribution: label 3 over ten
classes becomes `[0,0,0,1,0,0,0,0,0,0]`.  Stage 4 showed why you rarely build it,
since the nine zero terms contribute nothing and integer indexing gets the same
answer.

**Batch** [stage 2] — a set of examples processed together, `B` in the shape
notation.  **Full batch** is the entire training set at once, as in Stage 5.
**Mini-batch** [stage 8] is a small chunk, typically 32 to 512.  Batch size is
the main reason `mean` rather than `sum` appears in every loss in this course:
the mean keeps gradients the same size when the batch size changes, so a learning
rate tuned at one batch size still works at another.

**Epoch** [stage 8] — one full pass through the training data.  **Step /
iteration / update** is one parameter update from one mini-batch.  These get
mixed up constantly, and the ratio matters: 60,000 examples in batches of 100
means 600 steps per epoch, so "trained for 10 epochs" and "trained for 10 steps"
differ by a factor of 600.

**Shuffling** [stage 8] — reordering examples between epochs, so the model does
not learn the order.  Shuffle features and labels with the *same* permutation;
the bug where they come apart is on the teaching notes' list of mistakes worth
making once, because the resulting model trains smoothly to nonsense.

**Normalization** [stage 7] — rescaling inputs to a comfortable range, which for
this course is dividing pixels by 255 to land in `[0, 1]`.  It is not cosmetic:
Stage 4, Part B pointed out that raw `0..255` pixels against careless weights are
how you get a logit of 1,000 and a `nan`.  **Standardization** [beyond] is the
close cousin that subtracts the mean and divides by the standard deviation to get
mean 0 and variance 1.

**Training / validation / test split** [stage 7] — three disjoint sets with three
different jobs, and the rule is worth stating as a rule:

```text
training set     the parameters are fitted on it
validation set   you choose learning rate, width and epochs by looking at it
test set         you look once, at the end, and then change nothing
```

The test set estimates performance on data nobody has adapted to.  Every time you
adjust something after seeing a test number, that number becomes a little more
optimistic and a little less true.

**Data leakage** [beyond] — when information from validation or test reaches
training, by a shared preprocessing statistic, a duplicated example, or a split
made after some transformation that mixed the sets.  The symptom is a great score
that evaporates on genuinely new data.  Compute normalization constants on the
training set only.

**Class imbalance** [beyond] — when classes have very different frequencies.  The
reason Part 2 exists.  MNIST is nearly balanced, so this course never has to deal
with it.

**Data augmentation** [beyond] — inventing new training examples by transforming
existing ones: shifting an MNIST digit a pixel or two, rotating it slightly.
Cheaper than collecting data, and it encodes a real claim — a shifted 7 is still
a 7 — that the network otherwise has to learn from scratch.

**Distribution shift** [beyond] — the data you deployed on does not look like the
data you trained on.  A digit classifier trained on MNIST does poorly on
handwriting collected from a different population with different pens.  No metric
on your test set warns you about this, because your test set came from the same
place as your training set.

---

## Part 4 — Words about models

**Model** [stage 1] — the function from input to output, plus the numbers that
function uses.  Stage 1's model is `prediction = weight * x`, and it is a model in
the same sense as the final network.

**Parameter** [stage 1] — a number learned by training: every entry of `W1`, `b1`,
`W2`, `b2`.  **Hyperparameter** [stage 8] — a number *you* choose, that training
does not touch: learning rate, hidden width, batch size, epoch count, threshold.
The distinction is exact and worth keeping sharp: parameters are fitted on the
training set, hyperparameters are chosen on the validation set.  When someone
says a model "has 7 billion parameters," they mean the first kind.

**Weight** and **bias** [stage 2] — the multiplicative and additive parameters of
a layer.  The weight decides how much an input matters; the bias shifts the
output regardless of input.

**Affine layer / linear layer / fully connected layer / dense layer** [stage 9] —
four names for `Z = X @ W + b`.  Strictly, "linear" means without the bias and
"affine" means with it, and essentially nobody observes the distinction.  Torch
calls it `nn.Linear`.

**Logit** [stage 3] — an unbounded raw score, before any squashing.  The name
comes from the log-odds function, and the useful fact is Stage 4's: softmax reads
only the *differences* between logits in a row, so ten logits carry nine
independent numbers.

**Activation function** [stage 9] — the elementwise nonlinearity between layers.
Without one, stacked affine layers collapse into a single affine layer and depth
buys you nothing — which is the fact that makes XOR in Stage 10 the right test.

```text
ReLU(z)      = max(0, z)              [stage 9]   the default; cheap, no saturation
sigmoid(z)   = 1 / (1 + exp(-z))      [stage 3]   squashes to (0, 1)
softmax(z)   = exp(z) / sum(exp(z))   [stage 4]   a whole row to a distribution
tanh(z)                               [beyond]    squashes to (-1, 1)
leaky ReLU, GELU, SiLU                [beyond]    variants that leak a little at z < 0
```

Sigmoid and softmax also serve as *output* layers, which is a different job from
being an activation between layers.  Sigmoid is the two-class case of softmax,
not a separate idea: `softmax([z0, z1])[1] = sigmoid(z1 - z0)`, which is Stage 4,
question 8.

**Saturation** [stage 3] — when a squashing function is so far out on its flat
tail that its derivative is nearly zero and gradients stop flowing.  ReLU's
popularity is mostly this: it does not saturate for positive inputs.

**Dead ReLU** [stage 10] — a hidden unit whose pre-activation is negative for
every example in the data.  The ReLU mask zeroes its gradient, so it never
updates, so it stays dead.  This is a real failure mode, not a curiosity.

**Neuron / unit** [stage 9] — one column of a weight matrix plus its bias: a
weighted sum followed by an activation.  The biological analogy is thin enough
that "unit" is the better word.

**Layer, depth, width** [stage 9] — a layer is one transformation, depth is how
many, width is how many units in one.  Your final network has one hidden layer of
`H` units.

**MLP / multilayer perceptron / feedforward network / dense network** [stage 11] —
the architecture you are building: affine, activation, affine, softmax, with
every input connected to every unit.

**Architecture** [beyond] — the shape of the model: how many layers of what kind,
wired how.  Choosing it is a hyperparameter decision, and it is the part
frameworks do not do for you.

**Capacity** [stage 11] — how much a model can represent, roughly tracking its
parameter count.  Too little and it cannot fit the training data — which is what
Stage 11's tiny-batch overfit test checks first, since a model that cannot
memorize 64 examples has a bug or not enough capacity.  Too much and it can
memorize noise.

**Initialization** [stage 9] — the values parameters start at.  Not a detail: if
every hidden unit starts identical, every unit gets the same gradient forever and
64 units behave as one.  **He initialization** draws from
`Normal(0, sqrt(2 / fan_in))`, scaled to keep activations from shrinking or
exploding as they pass through layers.  Zero initialization is fine for Stage 3
and fatal from Stage 9 on, which is Stage 3's question 7.

**Inference** [stage 11] — running the model forward to get answers, with no
backward pass and no updates.  It needs no cached activations and no gradient
memory, so it is far cheaper than training.  Stage 4, Part H makes the sharpest
version of this point: at inference you can skip softmax entirely, because
`argmax(logits) == argmax(softmax(logits))`.

---

## Part 5 — Words about training

**Loss function / cost function / objective** [stage 1] — the single number
measuring how wrong the model is, which training exists to reduce.  Strictly,
"loss" is often per-example and "cost" the batch average; the words are used
interchangeably.  The ones you have built:

```text
mean squared error, MSE       [stage 2]   mean(0.5 * (prediction - target)^2)
binary cross-entropy, BCE     [stage 3]   -y log p - (1-y) log(1-p)
cross-entropy                 [stage 4]   -log(p of the true class)
```

**Cross-entropy** [stage 4] — the loss for classification, `-sum_x p(x) log q(x)`,
where the label picks the outcomes and the model sets the price.  It reads back
as a probability: `exp(-loss)` is the geometric mean probability the model gave
the right answer.  Related terms you will hear: **negative log-likelihood** is
the same quantity approached from statistics, and **KL divergence** differs from
cross-entropy by `H(p)`, a constant that does not depend on your parameters and
so changes nothing about which parameters gradient descent finds.

**Nat** and **bit** [stage 4] — units of the loss.  `np.log` is the natural
logarithm, so your losses are in nats; divide by `log(2)` for bits.

**Gradient** [stage 1] — the vector of partial derivatives of the loss with
respect to each parameter: which way is uphill, and how steeply.  A gradient
always has the same shape as the parameter it belongs to, which this course calls
its most useful debugging check.

**Gradient descent** [stage 1] — `parameter -= learning_rate * gradient`.  The
whole algorithm.  Variants by how much data each step uses:

```text
batch / full-batch gradient descent   [stage 5]   every example per step
stochastic gradient descent, SGD      [stage 8]   one mini-batch per step
```

"SGD" in practice always means the mini-batch version.  The noise it adds is not
purely a cost: steps are far cheaper, so you take many more of them per pass
through the data.

**Learning rate** [stage 1] — how far to move along the negative gradient.  The
hyperparameter that most often ruins a run.  Too small and training crawls; too
large and the loss oscillates or becomes `nan` — Stage 4, Part B named a slightly
oversized learning rate as the most common way logits reach 1,000 in real
training.  A **learning-rate schedule** [beyond] lowers it over time, to take
large steps early and settle later.

**Backpropagation** [stage 10] — the chain rule applied in reverse data-flow
order, which computes the derivative of one scalar loss with respect to all
parameters at a cost comparable to a single forward pass.  It is not a learning
algorithm; it is how gradient descent gets its gradients.

**Forward pass / backward pass** [stage 9, stage 10] — inputs to loss, then loss
back to gradients.  The **cache** the forward pass stores exists because the
backward pass needs those intermediate values, and holding them is a deliberate
trade of memory for recomputation.

**Automatic differentiation / autograd** [beyond] — what a framework does instead
of making you write Stage 10.  Each forward operation has a local derivative
rule, and reverse-mode AD chains them according to data dependence.  The
mechanism is exactly what you are implementing by hand; the framework only
bookkeeps it.

**Gradient checking** [stage 5] — comparing analytic gradients against central
finite differences, `(L(t+eps) - L(t-eps)) / (2*eps)`, in float64 on a tiny model.
It is a debugging oracle and never a training method: it costs two forward passes
per parameter.  Compare relative error, not just absolute.

**Optimizer** [beyond] — the rule that turns gradients into updates.  Plain
gradient descent is the simplest.  **Momentum** accumulates a velocity so
consistent directions build speed.  **Adam** keeps a running estimate of each
parameter's gradient scale and adapts a per-parameter step size; it is the
default in most modern code.  Every one of these adds *state* that has to be
saved in a checkpoint alongside the parameters.

**Vanishing / exploding gradients** [beyond] — in deep networks, gradients
shrinking toward zero or blowing up as they multiply back through many layers.
Initialization scaling, ReLU, normalization layers, and residual connections are
all partly responses to this.  **Gradient clipping** rescales gradients above a
chosen norm.

**Convergence** [stage 2] — training reaching the point where the loss stops
improving meaningfully.  Not the same as being correct, and not the same as
reaching zero loss: Stage 3's classifier hits 100 percent accuracy while its loss
keeps falling for hundreds of steps, and the loss never reaches zero at all.

**Local minimum / saddle point** [beyond] — points where the gradient is zero
without being the global best.  Textbooks worry about them more than practice
does; in high dimensions the practical problems are more often long flat regions
and badly scaled steps.

**Training curve / learning curve** [stage 11] — loss and accuracy plotted
against epoch, for train and validation together.  The single most informative
diagnostic you get, and Stage 11, Part C lists the patterns to read off it.

---

## Part 6 — Words about generalization

**Generalization** — performing well on data the model was not fitted on.  The
actual goal.  Training loss measures memorization; validation measures this.

**Overfitting** [stage 11] — fitting the training data, including its noise,
closely enough that validation performance stops improving or gets worse.  The
signature in the curves is training loss falling while validation loss rises.
The **generalization gap** is the distance between the two.

**Underfitting** [stage 11] — the model is too limited, or trained too briefly,
to fit even the training data.  Both losses stay high and flat.  The fix is the
opposite of overfitting's: more capacity, more training, better optimization.

**Bias–variance tradeoff** [beyond] — the classical framing of that tension.
Bias is error from a model too simple to represent the truth; variance is error
from sensitivity to the particular training sample.  Useful as vocabulary; less
predictive for large networks than it was for smaller models.

**Regularization** [beyond] — anything that constrains a model to improve
generalization at some cost in training fit:

```text
L2 / weight decay    add 0.5 * lambda * sum(W^2) to the loss, so dW += lambda * W
L1                   penalize sum(|W|); pushes weights to exactly zero
dropout              randomly zero a fraction of activations during training only
early stopping       stop when validation loss stops improving
data augmentation    described in Part 3
```

Dropout is the one that forces a real structural change: the network must behave
differently during training than during inference, which is where a framework's
`model.train()` and `model.eval()` modes come from.

**Memorization** [stage 11] — fitting specific examples rather than a pattern.
Usually the failure mode, but Stage 11's tiny-batch overfit test *wants* it,
because a model that cannot memorize 64 examples cannot fit anything and the bug
is in your code rather than your data.

**Ablation** [beyond] — removing one piece to see how much it was contributing.
The honest way to find out whether a change helped.

---

## Part 7 — Kinds of learning problem

**Supervised learning** [stage 1] — learning from labeled examples.  Everything in
this course.

**Unsupervised learning** [beyond] — finding structure in unlabeled data:
clustering, dimensionality reduction.

**Self-supervised learning** [beyond] — manufacturing labels from the data's own
structure, such as predicting the next word in a sentence.  No human labels
needed, which is why it scales, and it is how language models are trained.

**Reinforcement learning** [beyond] — learning from rewards for actions taken over
time, rather than from labeled correct answers.  A different setting, not a
different network.

**Classification** [stage 3] — predicting which of a finite set of classes.
**Regression** [stage 2] — predicting a continuous number.  Stage 2 is regression
and Stage 3 onward is classification, which is why the loss changes and the
training loop does not.

**Binary / multi-class / multi-label** [stage 4] — two classes, one of many
classes, or any number of independent yes/no answers at once.  Stage 4, Part A is
the argument for why multi-class wants one softmax and multi-label wants
independent sigmoids: softmax makes the classes compete for a fixed budget of
probability, and for multi-label there is nothing to compete over.

**Transfer learning / fine-tuning** [beyond] — starting from a model trained on
one task and continuing training on your own, usually much smaller, dataset.  The
dominant way people use large models today.

---

## Part 8 — Words from outside this course

You will hear these constantly.  Each gets one honest sentence and, where there
is one, its connection to what you have built.

**Artificial intelligence / machine learning / deep learning** — nested, from
widest to narrowest: AI is the ambition, machine learning is the approach of
fitting parameters to data, deep learning is machine learning with many-layer
neural networks.  Your Stage 11 network is deep learning at its smallest
plausible size.

**Neural network** — a stack of parameterized transformations with nonlinearities
between them.  You are building one, and there is no additional secret in the
large ones beyond scale, architecture, and data.

**Tensor** — an array with more than two dimensions, and in framework usage, any
array.  `np.ndarray` is a tensor; Torch's version adds device placement and
gradient tracking.

**GPU / CUDA / accelerator** — hardware that does thousands of multiply-adds in
parallel, which is what `@` needs.  This is why matrix multiplication is the
unit of work in deep learning, and why your NumPy code on a CPU trains MNIST in
minutes while a large model needs a cluster for months.

**Framework** — PyTorch, TensorFlow, JAX.  They supply autograd, GPU kernels,
optimizers, data loaders, and checkpointing.  The point of this course is that
none of that is mysterious once you have written the small version by hand.

**CNN / convolutional network** — a network that slides small shared filters over
an image rather than connecting every pixel to every unit.  It builds in the
knowledge that nearby pixels are related and that a stroke means the same thing
wherever it appears, which your dense network has to learn from scratch.  This is
the natural next step after Stage 11, and it is where MNIST accuracy above 99
percent comes from.

**RNN / LSTM** — networks that process sequences one element at a time while
carrying state.  Largely displaced by transformers for language.

**Transformer** — the architecture behind essentially all current large language
models.  Its distinguishing part is **attention**, which lets every position in a
sequence look at every other position and decide what is relevant, rather than
processing in fixed order.

**Token / embedding** — text is chopped into tokens (roughly word pieces), and
each token id is mapped to a learned vector, its embedding.  Embeddings are just
rows of a weight matrix, learned by the same gradient descent you wrote in
Stage 1.

**LLM / large language model** — a transformer trained by self-supervision to
predict the next token, at enormous scale.  Here is the connection worth
carrying: its final layer produces one logit per token in a vocabulary of tens of
thousands, applies softmax, and takes cross-entropy against the actual next
token.  That is Stage 4, unchanged, with `C = 50000` instead of `C = 10`.  The
`log(C)` landmark still works: an untrained model's loss starts near
`log(vocabulary size)`.

**Pretraining / fine-tuning / alignment** — the expensive general-purpose training
run, then cheaper adaptation to a specific task, then further training to make
the model's behavior useful and safe.

**Batch normalization / layer normalization** — layers that rescale activations
using batch or per-example statistics, to keep the scale of numbers flowing
through a deep network under control.  A descendant of the same concern that
motivates He initialization in Stage 9.

**Embedding space / latent space / representation** — the vector of activations a
network computes partway through, treated as a learned description of the input.
Your `H1` in Stage 9 is a 64-number representation of a 784-pixel image, and
looking at what those 64 numbers respond to is a real experiment you can run.

**Benchmark / SOTA** — a standard dataset and metric used to compare models, and
"state of the art," the best published score on it.  MNIST is a benchmark, and it
has been solved long enough that its remaining value is entirely pedagogical.

**Checkpoint** — the saved parameters, and any optimizer state, needed to resume
or deploy.  Stage 11, Part G writes one with `np.savez_compressed`.

**Hyperparameter search / grid search / random search** — trying many
hyperparameter combinations and keeping the best by validation score.  Random
search usually beats grid search for the same budget, because most
hyperparameters do not matter equally.

**MLOps / model serving / drift monitoring** — the engineering around a trained
model: deploying it, watching its inputs change over time, retraining.  Where
most industrial effort actually goes, and entirely outside this course.

---

## Part 9 — Numbers worth memorizing

The course keeps insisting on landmark numbers because they are the cheapest
diagnostics you will ever have.  Collected in one place:

```text
0.6931 = log(2)      two-class loss before learning anything      [stage 3]
2.3026 = log(10)     ten-class loss before learning anything      [stage 4]
10%                  accuracy of guessing on ten balanced classes
0.5                  the default decision threshold, and a choice [stage 3]
784 = 28 * 28        MNIST features after flattening
255                  the largest raw pixel value, hence /255      [stage 7]
60,000 / 10,000      MNIST training and test set sizes
38                   the logit where a float64 probability rounds to exactly 1.0
746                  how far below a row's max an exponential underflows to 0.0
7,850                parameters in the Stage 8 linear model, 784*10 + 10
50,890               parameters in a 64-unit Stage 11 MLP, 784*64 + 64 + 64*10 + 10
exp(-loss)           the average probability given to the correct class
```

Two ways to use these.  A ten-class model whose step-0 loss is not near `2.303` is
broken before it has seen any data, so check that before debugging anything else.
And a loss of `0.10` means about 90 percent average confidence in the right
answer, which turns an abstract number into a sentence.

Rough accuracy expectations on MNIST, worth having as a sanity range rather than
a target.  These depend on training choices and are not promises:

```text
guessing                      about 10%
Stage 8 linear softmax        about 90-92%
Stage 11 MLP, well trained    about 95-98%
a convolutional network       about 99% and above
```

If your Stage 11 network lands at 60 percent, the problem is a bug or a
hyperparameter, not the limits of the architecture.  That is what the range is
for.

---

## Part 10 — False friends

Words that do not mean what they sound like, or that mean two things.  This
section is short because each entry is a mistake worth not making twice.

**Precision** has two unrelated meanings, and you will meet both in this
project.  In Part 2 it is `TP / (TP + FP)`, a property of a classifier's
decisions.  In "floating-point precision," it is how many digits a number type
can represent — `float32` versus `float64`, the subject of Stage 4's numerical
troubles.  Nothing connects them but the word.  A science class may also have
taught you a third sense, where accuracy means closeness to the truth and
precision means repeatability; that pair is from measurement, not from machine
learning, and neither one matches the definitions in Part 2.

**Accuracy** in Part 2 is one specific fraction, not a general word for how good
the model is.  Say "accuracy" when you mean that fraction.

**Loss is not error rate.**  Loss is the continuous quantity you differentiate;
error rate is `1 - accuracy`, a fraction of decisions.  They usually move
together, and Stage 3 shows they need not.

**Prediction is not probability.**  A prediction is a class, after a decision.  A
probability is a confidence, before one.  Stage 3's three levels exist because
mixing these up is the most common Stage 3 bug.

**Validation is not test.**  Both are held out; only one may be looked at
repeatedly.  Some fields swap the two names, so when reading a paper, check which
one they tuned on.

**Bias** means two unrelated things: the `b` in `X @ W + b`, and the statistical
sense in "bias–variance."  Neither means social bias, which is a third, real, and
much-discussed thing.  Context is your only guide.

**A parameter is not a hyperparameter.**  Learned versus chosen.  See Part 4.

**Epoch is not step.**  See Part 3.  A factor of hundreds hides here.

**Normalize** is used for at least three operations: scaling to `[0, 1]`,
standardizing to mean 0 and variance 1, and dividing a vector by its sum so it
becomes a distribution — which is what the denominator in softmax does.  Ask
which one is meant.

**Deep** describes many layers, not quality.  Your one-hidden-layer network is
shallow, and it will still classify handwriting.

**Learning** is curve fitting by gradient descent.  Keeping that translation in
mind will save you from a great deal of overexcited writing about what these
systems are doing.

---

## Questions to answer

In the style of the stages.  These check Part 2, since that is the part with real
content in it rather than definitions.

1. A model that never predicts 8 is 90 percent accurate on the is-it-an-8
   problem.  Compute its precision, recall, and F1, and say which metric exposes
   it most clearly.
2. Precision and recall have the same numerator.  State the rule for which
   denominator belongs to which, without looking it up.
3. You raise the decision threshold from `0.5` to `0.9`.  Which of the four
   counts can go up, which can go down, and what happens to precision and recall?
4. Why can you change the precision–recall trade-off without retraining, but not
   change the accuracy of a fixed threshold without retraining?
5. Give one problem where a false negative is much worse than a false positive,
   and one where the reverse holds.  Say which metric you would report for each.
6. Why is the harmonic mean used for F1 rather than the arithmetic mean?  Support
   your answer with precision `1.0` and recall `0.01`.
7. Stage 11's confusion matrix has true digits as rows and predicted digits as
   columns.  Write the expressions for the recall and the precision of digit 9,
   and for overall accuracy, in terms of that matrix.
8. On a single-label ten-class problem, micro-averaged precision, micro-averaged
   recall, micro-averaged F1, and accuracy are all the same number.  Explain why
   by tracking what one misclassified image contributes to each count.
9. A model reaches 97 percent accuracy.  Name two different things that could be
   true about its per-class recalls, one benign and one worth fixing.
10. Why is no metric in Part 2 usable as a training loss?  Answer in terms of the
    four levels in Part 1.
11. Your model is 95 percent accurate and badly calibrated.  Using Stage 4's
    temperature experiment, explain how both can be true at once.
