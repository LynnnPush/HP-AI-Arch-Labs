import os

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from jax import random
from tensorflow.keras.datasets import mnist
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(jax.devices())

# loaing and normalizing the data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize
x_train, x_test = x_train.reshape(-1, 784), x_test.reshape(-1, 784)  # Flatten

# pre-move to device so the per-step slicing stays on GPU
x_train_j = jnp.array(x_train)
y_train_j = jnp.array(y_train)
x_test_j = jnp.array(x_test)
y_test_j = jnp.array(y_test)


class MLP(nn.Module):
    dropout_rate: float = 0.3

    @nn.compact
    def __call__(self, x: jax.Array, train: bool) -> jax.Array:
        x = nn.Dense(512)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
        x = nn.Dense(512)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
        x = nn.Dense(10)(x)
        return x


model = MLP()
optimizer = optax.adamw(learning_rate=1e-3, weight_decay=1e-4)


@jax.jit
def loss(params: dict, x: jax.Array, y: jax.Array, do_rng: jax.Array) -> jax.Array:
    logits = model.apply(
        {"params": params}, x, train=True, rngs={"dropout": do_rng}
    )  # foward pass of the model, calling the __call__ function and returning the output
    return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, y))


@jax.jit
def predict(params: dict, x: jax.Array) -> jax.Array:
    logits = model.apply({"params": params}, x, train=False)
    return jnp.argmax(logits, axis=-1)


@jax.jit
def accuracy(params: dict, x: jax.Array, y: jax.Array) -> jax.Array:
    predictions = predict(params, x)
    return 100 * jnp.mean(predictions == y)


@jax.jit
def training_step(
    params: dict, opt_state: tuple, x: jax.Array, y: jax.Array, do_rng: jax.Array
):
    loss_value, grads = jax.value_and_grad(loss)(params, x, y, do_rng)

    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    return params, opt_state, loss_value


# multi-seed sweep configuration
SEEDS = (10, 42, 2024)
NUM_EPOCHS = 10
BATCH_SIZE = 128

results = {}

for seed in SEEDS:
    print("\n" + "=" * 70)
    print(f"Run with seed = {seed}  (epochs = {NUM_EPOCHS}, batch = {BATCH_SIZE})")
    print("=" * 70)

    # seed controls both weight init and the dropout RNG stream
    key = random.PRNGKey(seed)
    init_key, key = random.split(key)
    params = model.init(init_key, jnp.ones([1, 784]), train=True)["params"]
    opt_state = optimizer.init(params)

    print("Untrained model accuracy:", accuracy(params, x_test_j, y_test_j))

    epoch_test_acc = []
    for epoch in range(NUM_EPOCHS):
        for i in (pbar := tqdm(range(0, len(x_train), BATCH_SIZE), desc=f"Seed {seed} Epoch {epoch}")):
            key, subkey = random.split(key)

            params, opt_state, train_loss = training_step(
                params,
                opt_state,
                x_train_j[i : i + BATCH_SIZE],
                y_train_j[i : i + BATCH_SIZE],
                subkey,
            )

            if i % (BATCH_SIZE * 5) == 0:
                pbar.set_postfix(
                    {
                        "TRAIN LOSS": float(loss(params, x_train_j[:1000], y_train_j[:1000], subkey)),
                        "TEST ACC": float(accuracy(params, x_test_j, y_test_j)),
                    }
                )

        # end-of-epoch test accuracy for the summary table
        epoch_acc = float(accuracy(params, x_test_j, y_test_j))
        epoch_test_acc.append(epoch_acc)
        print(f"  end of epoch {epoch}: test accuracy = {epoch_acc:.2f}%")

    test_accuracy = float(accuracy(params, x_test_j, y_test_j))
    train_accuracy = float(accuracy(params, x_train_j, y_train_j))
    print(f"Seed {seed}  Test: {test_accuracy:.2f}%  Train: {train_accuracy:.2f}%")

    results[seed] = {
        "epoch_test_acc": epoch_test_acc,
        "final_test_acc": test_accuracy,
        "final_train_acc": train_accuracy,
    }


# summary table for easy reporting
print("\n" + "=" * 70)
print(f"SUMMARY: test accuracy per epoch across {len(SEEDS)} seeds")
print("=" * 70)
header = "Epoch | " + " | ".join(f"seed={s:<6}" for s in SEEDS) + " | mean"
print(header)
print("-" * len(header))
for epoch in range(NUM_EPOCHS):
    vals = [results[s]["epoch_test_acc"][epoch] for s in SEEDS]
    mean_v = sum(vals) / len(vals)
    row = f"{epoch:>5} | " + " | ".join(f"{v:>10.2f}%" for v in vals) + f" | {mean_v:>5.2f}%"
    print(row)

print("\nFinal results:")
print(f"{'Seed':<8} {'Train acc':>12} {'Test acc':>12}")
for s in SEEDS:
    print(f"{s:<8} {results[s]['final_train_acc']:>11.2f}% {results[s]['final_test_acc']:>11.2f}%")

# aggregate stats across seeds — quantifies run-to-run variability
final_tests = [results[s]["final_test_acc"] for s in SEEDS]
final_trains = [results[s]["final_train_acc"] for s in SEEDS]
mean_test = sum(final_tests) / len(final_tests)
mean_train = sum(final_trains) / len(final_trains)
spread_test = max(final_tests) - min(final_tests)
spread_train = max(final_trains) - min(final_trains)
print(f"\nMean final test  accuracy: {mean_test:.2f}%  (spread = {spread_test:.2f} pp)")
print(f"Mean final train accuracy: {mean_train:.2f}%  (spread = {spread_train:.2f} pp)")
