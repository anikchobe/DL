## Stage 8 — Train a linear MNIST baseline

### Goal

Apply the Stage 5 multiclass classifier to real images and introduce mini-batch
stochastic gradient descent, epochs, validation metrics, and model interpretation.

The model is:

```text
X:      (B, 784)
W:      (784, 10)
b:      (10,)
logits: (B, 10)

logits = X @ W + b
```

### Part A: initialize the model

Create zero-valued `W` and `b`.  Unlike hidden units, the ten output columns do not
remain permanently identical: labels create different gradients for each class.
You may also compare with small random initialization.

### Part B: mini-batches

Write a generator:

```python
def iterate_minibatches(
    x: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    *,
    rng: np.random.Generator,
    shuffle: bool = True,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    ...
```

Requirements:

- shuffle once at the beginning of each training epoch;
- do not shuffle validation or test evaluation;
- yield the final short batch rather than dropping it;
- preserve feature/label alignment by indexing both with the same indices.

### Part C: one update

For one batch:

```text
logits -> loss and dlogits -> dW and db -> parameter update
```

Keep the four phases in separate functions at first.  Add assertions:

```python
assert gradient_weights.shape == model.weights.shape
assert gradient_bias.shape == model.bias.shape
```

### Part D: an epoch

One epoch means that every training example has appeared in one mini-batch.  After
all updates in an epoch, compute:

```text
training loss and accuracy
validation loss and accuracy
```

Do not average batch percentages naïvely if the final batch is smaller.  Accumulate
counts or weight batch metrics by batch size.

### Part E: quick run, then full run

Start with a subset and a few epochs.  The purpose of the first run is to catch
shape, loss, and update bugs quickly.  Only then use all training examples.

Suggested command after implementing the CLI:

```bash
python solution8.py --quick
```

Record:

```text
epoch | train loss | train accuracy | validation loss | validation accuracy
```

The precise values depend on initialization, split, and hyperparameters.  The
important checks are that loss falls, accuracy rises well above chance, and train
and validation behavior remain plausible.

### Part F: visualize the learned weights

Each class has one column of `W` containing 784 values.  Reshape:

```python
model.weights[:, digit].reshape(28, 28)
```

Display all ten.  Positive and negative values indicate pixels that push the score
toward or away from a class.  Do not call these literal average digits; they are
discriminative weight templates influenced by competition among classes.

### Minimum tests

Use a small synthetic three-class dataset so tests do not need MNIST:

- mini-batches cover every example exactly once when `shuffle=False`;
- the last short batch is present;
- one update changes parameters;
- several epochs reduce loss and reach high toy accuracy;
- evaluation does not modify the model;
- a fixed seed gives a reproducible training history.

### Questions to answer

1. How is mini-batch SGD different from full-batch gradient descent?
2. Why shuffle between epochs?
3. What is noisy about a mini-batch gradient?
4. What work scales with `B * 784 * 10`?
5. Why can the linear classifier recognize digits at all despite having no hidden
   layer?
6. What kinds of differences between handwritten digits are difficult for a
   single linear boundary?
7. Why choose hyperparameters from validation performance rather than training
   performance?

### Stretch

Train with batch sizes 1, 32, 128, and the full training set.  Compare time per
epoch, smoothness of the loss, and validation accuracy.

