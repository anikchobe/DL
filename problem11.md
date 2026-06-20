## Stage 11 — Train, save, and diagnose the NumPy MNIST network

### Goal

Combine all earlier pieces into an end-to-end trainer, then use diagnostics to
distinguish a working model from a merely running program.

The final model is:

```text
784 inputs -> H ReLU hidden units -> 10 logits
```

### Part A: tiny-batch overfit test

Before a full training run, select roughly 64 examples and repeatedly train on that
same set.

Write:

```python
def overfit_tiny_batch(...):
    ...
```

A network with enough hidden units should be able to memorize a tiny set.  Failure
to do so is a strong sign of a bug, an unsuitable learning rate, or a capacity
problem.

This test answers a different question from validation accuracy:

```text
Can the implementation fit data at all?
```

### Part B: mini-batch MLP training

Reuse the Stage 8 mini-batch generator and Stage 10 gradients.  For every batch:

```text
forward and loss
backward gradients
W1 update
b1 update
W2 update
b2 update
```

After each epoch, evaluate train and validation metrics without updating anything.
Store a history record.

Start with:

```text
training examples: 15,000
validation examples: 2,000
hidden units: 64
epochs: 5
```

Then increase the dataset and epoch count after the pipeline behaves correctly.

### Part C: training curves

Plot two separate figures:

```text
train and validation loss versus epoch
train and validation accuracy versus epoch
```

Interpret patterns:

- both losses high and flat: model is not learning.  This is
  [underfitting](GLOSSARY.md#part-6--words-about-generalization), and its fix is
  the opposite of the next line's;
- training improves but validation does not:
  [overfitting](GLOSSARY.md#part-6--words-about-generalization) or a mismatch
  between the splits.  Stage 11 only shows you this shape; Extension 1 adds L2
  and early stopping, which are the two cheapest things to do about it;
- loss becomes `nan`: numerical instability or excessive learning rate;
- accuracy stays near ten percent: likely random guessing or a serious bug.  Ten
  percent is not a low score, it is the score of no model at all — see
  [baselines](GLOSSARY.md#baselines-stage-11);
- training accuracy high but tiny-batch test failed earlier: investigate
  inconsistent code paths.

### Part D: final test evaluation

Only after choosing hyperparameters from validation results, evaluate the original
10,000-example test set.  Report both loss and accuracy.

Do not alter the model based on this test result and then continue calling it a
clean final test.

### Part E: confusion matrix

Implement:

```python
def confusion_matrix(
    labels: np.ndarray,
    predictions: np.ndarray,
    class_count: int = 10,
) -> np.ndarray:
    ...
```

Rows represent true digits and columns predicted digits.  A concise NumPy update is:

```python
np.add.at(matrix, (labels, predictions), 1)
```

Rows and columns turn into per-class metrics directly:
[recall of digit c](GLOSSARY.md#ten-classes-read-them-off-the-confusion-matrix-stage-11)
is `M[c, c]` divided by the sum of row `c`, and its precision is `M[c, c]`
divided by the sum of column `c`.  Compute both for all ten digits.  A model at
97 percent overall can be at 93 percent on one digit, and the accuracy number
cannot tell you which.

Print the most common off-diagonal pairs:

```text
true 5 predicted 3
true 9 predicted 4
...
```

The actual pairs will depend on training.

### Part F: inspect mistakes

Find:

```python
wrong = np.flatnonzero(predictions != labels)
```

Use the Stage 7 grid to display true and predicted labels.  Categorize a few
mistakes:

- genuinely ambiguous handwriting;
- unusual stroke placement;
- likely linear/MLP representation limitation;
- possible label surprise;
- confident versus uncertain mistakes.

For an uncertain prediction, print the top three class probabilities.

### Part G: save and reload

Store the four arrays with:

```python
np.savez_compressed(
    path,
    weights1=...,
    bias1=...,
    weights2=...,
    bias2=...,
)
```

Load with `allow_pickle=False`.  Verify that predictions before and after reload are
identical for a sample batch.

### Minimum tests

The automated unit tests should still use small synthetic data:

- trainer reduces loss on an easy three-class problem;
- confusion-matrix entries sum to the number of examples;
- save/load preserves every parameter array and prediction;
- evaluation does not mutate parameters;
- tiny-batch overfit reaches very high training accuracy;
- a fixed seed gives reproducible initialization and batch order.

The real MNIST run is an integration test rather than the only test.

### Questions to answer

1. Why is tiny-batch memorization a useful debugging step but not evidence of
   generalization?
2. What is the difference between an epoch and an update step?
3. Why can validation loss worsen before validation accuracy visibly changes?
4. What does a confusion matrix reveal that one accuracy number hides?
5. Why save raw arrays rather than serializing arbitrary Python objects?
6. Which portions of this project would Torch normally supply?
7. Which portions would remain the user's responsibility even with Torch?
8. How would inference differ from training in memory use and required operations?

### Stretch

Add top-k accuracy and confidence
[calibration](GLOSSARY.md#two-more-metrics-worth-knowing-stage-11-stretch) bins.
A prediction can be correct while poorly calibrated, or wrong while
appropriately uncertain.  Extension 4 is this stretch done properly: reliability
diagrams, expected calibration error, and a temperature fitted on the validation
set — which changes every probability the model reports and not one of its
decisions.

The other extensions all start from a run you have now made.  Extension 1
regularizes the overfitting in Part C, Extension 2 replaces the fixed learning
rate, Extension 5 builds the imbalanced problem MNIST never gives you, and
Extension 7 replaces the architecture while reusing everything else in this
file.

