"Logit" is a contraction of "logistic unit," coined by the statistician Joseph Berkson in 1944 (Application of the Logistic Function to Bio-Assay, JASA). He built it by analogy with "probit" = "probability unit," which Chester Bliss had coined in 1934 for the corresponding quantity under the normal curve. Berkson said as much in the paper — he was naming his function after Bliss's.

And it's not just a name for "raw score." A logit is a specific quantity with a specific meaning:

logit(p) = ln( p / (1 − p) )        the log of the odds

It's the inverse of the sigmoid

This is the part that makes the code make sense. Sigmoid and logit are the same function read in opposite directions:

sigmoid : ℝ → (0,1)        squashes a score into a probability
logit   : (0,1) → ℝ        stretches a probability back into a score

  p       odds = p/(1−p)     logit = ln(odds)     sigmoid(logit)
 0.01          0.010            −4.5951             0.01
 0.25          0.333            −1.0986             0.25
 0.50          1.000             0.0000             0.50
 0.75          3.000            +1.0986             0.75
 0.99         99.000            +4.5951             0.99

So when Stage 3 writes logits = X @ weights + bias and then p = sigmoid(logits), the name is literally accurate: that quantity is logit(p). The model computes log-odds, and sigmoid converts back to a probability. You can derive it in three lines — set p = 1/(1+e^−z), solve for z, and you get z = ln(p/(1−p)).

Two things in the code fall straight out of this, and they're worth showing your student:

- The decision boundary is at logit 0 — not an arbitrary convention. Odds of 1:1 means log-odds of 0 means p = 0.5. Even money.
- The table is antisymmetric. logit(0.99) = +4.5951 and logit(0.01) = −4.5951. Swapping the two classes just flips the sign, because logit(1−p) = −logit(p). That symmetry is exactly what you want from a linear model that treats both classes alike.

Why log odds, rather than odds or probability?

Because it's the scale on which a linear model can work.

- Probability is boxed in to [0,1]. X @ w + b produces any real number, so it can't be a probability directly.
- Odds are unbounded above but not below — they live in (0, ∞), and are multiplicatively lopsided: "twice as likely" is ×2, "half as likely" is ×0.5. Not symmetric.
- Log-odds cover all of ℝ, symmetrically, and — this is the real point — turn multiplication into addition. Evidence combines multiplicatively in odds space, so it combines additively in log-odds space.

That last property is why the model is a sum of weighted features. A linear model adds things up; log-odds is the scale on which adding up is the correct operation. Logistic regression isn't "linear model + a squashing hack" — it's a linear model of the log-odds.

The payoff: your weights become readable

Since logit is measured in log-odds, each weight has a real-world unit: log-odds per unit of that feature. Exponentiate it and you get an odds ratio. From your trained Stage 3 model:

weights = [1.6686, 1.6667]
exp(1.6686) = 5.305

Moving one unit along feature 0 multiplies the odds of class 1 by 5.3. That's a genuine, quotable interpretation of a learned parameter — and it's the standard way logistic-regression coefficients are reported in medicine and epidemiology ("the odds ratio was 5.3"). Same math, same vocabulary, invented for exactly this purpose in bioassay work in the 1930s–40s.

One caveat for Stage 4

In modern deep learning the word has drifted: "logits" now often just means whatever raw numbers you feed into a sigmoid or softmax, with the log-odds reading held loosely. For softmax with 10 classes, the strict interpretation needs care:

softmax([1,2,3])       = [0.0900, 0.2447, 0.6652]
softmax([1,2,3] + 100) = [0.0900, 0.2447, 0.6652]      identical

An individual softmax logit isn't a log-odds of anything — it's only deant. What is meaningful is the difference: z_i − z_j = ln(p_i / p_j),
the log-odds ratio between two classes. Verified above: z₂ − z₁ = 1.0 a

That shift-invariance is worth flagging now, because it's the same factle softmax valid. Subtracting the row maximum isn't an approximation —
it's exploiting the fact that logits are only defined up to a shift. Bee numerical trick turn out to be the same observation.

Two naming traps

- "Logistic" here has nothing to do with logistics (supply chains, movim the logistic curve, named by Pierre-François Verhulst in the 1840s for
population growth. Why he chose that word is genuinely unsettled among m "log-", plausibly from French logistique meaning calculation. Berkson
inherited the adjective; he only coined the noun.
- "Logit" is not "log it." It's logistic + -it, pronounced like "probitfor "unit of," in the same family as "probit" and, later, "tobit."

If you want a one-sentence version for the student: a logit is a probabodds, so that it lives on the whole number line and evidence addsinstead of multiplies — which is precisely what a linear model needs.
