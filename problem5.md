## Stage 5 — Backpropagate through a multiclass linear classifier

### Goal

Connect the Stage 4 loss gradient to learned weight and bias arrays, verify the
result numerically, and train a small three-class model.

Generate three linearly separable clusters in two dimensions.  Use:

```text
X:       (B, 2)
W:       (2, 3)
b:       (3,)
logits:  (B, 3)
```

### Part A: forward pass

Implement:

```python
def logits(x: np.ndarray, parameters: LinearParameters) -> np.ndarray:
    return x @ parameters.weights + parameters.bias
```

Add explicit shape checks before the arithmetic.

### Part B: backward pass

Given:

```text
dlogits: (B, 3)
```

derive and implement:

```text
dW = X.T @ dlogits
db = sum(dlogits, axis=0)
```

Explain the meaning of one entry:

```text
dW[feature_index, class_index]
```

as a sum of contributions from all examples.

### Part C: gradient checking

Write a checker that chooses several entries from both `W` and `b`.  For each
entry:

1. save the original value;
2. add `epsilon` and compute the loss;
3. subtract `epsilon` and compute the loss;
4. restore the original value;
5. compare the central difference with the analytic gradient.

Use a relative error such as:

```text
abs(analytic - numerical) / max(1, abs(analytic), abs(numerical))
```

Do this on a tiny batch.  A gradient checker is deliberately slow and should not
run inside the training loop.

### Part D: train

Use full-batch gradient descent on the three clusters.  Record loss and accuracy.
The classifier should learn one score plane per class.

### Minimum tests

- `W` and `dW` have the same shape.
- `b` and `db` have the same shape.
- The numerical and analytic gradients agree for sampled entries.
- Loss decreases during training.
- Accuracy becomes high on the separable toy data.
- The same random seed produces the same dataset and initialization.

### Questions to answer

1. Where does the transpose in `X.T @ dlogits` come from?
2. Why does the bias gradient sum over the batch dimension?
3. What would happen if `dlogits` were not divided by batch size?
4. Does a passing gradient check prove that the model will train successfully?
5. Which part of this stage is already “backpropagation” even though there is only
   one layer?

### Stretch

Plot the class regions over a 2-D grid.  Then modify the data so the classes are
not linearly separable and observe the limit of the model.

