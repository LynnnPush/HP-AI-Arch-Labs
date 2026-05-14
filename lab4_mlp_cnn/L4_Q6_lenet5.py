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

# load + normalize MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# LeNet-5 expects 32x32 input — pad MNIST's 28x28 with 2 pixels of zeros on each side
x_train = jnp.pad(x_train, ((0, 0), (2, 2), (2, 2)))
x_test = jnp.pad(x_test, ((0, 0), (2, 2), (2, 2)))

# Flax conv layers use NHWC — add a trailing channel dim
x_train_j = x_train[..., None]
x_test_j = x_test[..., None]
y_train_j = jnp.array(y_train)
y_test_j = jnp.array(y_test)
print("train shape:", x_train_j.shape, "test shape:", x_test_j.shape)


class LeNet5(nn.Module):
    """LeNet-5 (LeCun 1998) with C3 fully connected across S2 channels
    (the standard modern simplification — symmetry breaking is handled by
    random init, so the original sparse connection table is omitted)."""

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        # C1: 6 feature maps, 5x5 kernel, valid padding → 28x28x6
        x = nn.Conv(features=6, kernel_size=(5, 5), padding="VALID")(x)
        x = nn.tanh(x)
        # S2: 2x2 average pool, stride 2 → 14x14x6
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))

        # C3: 16 feature maps, 5x5 → 10x10x16
        x = nn.Conv(features=16, kernel_size=(5, 5), padding="VALID")(x)
        x = nn.tanh(x)
        # S4: 2x2 avg pool, stride 2 → 5x5x16
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))

        # C5: 120 feature maps, 5x5 → 1x1x120, then flatten to 120
        x = nn.Conv(features=120, kernel_size=(5, 5), padding="VALID")(x)
        x = nn.tanh(x)
        x = x.reshape((x.shape[0], -1))

        # F6: 84-unit dense
        x = nn.Dense(features=84)(x)
        x = nn.tanh(x)
        # Output: 10-class logits
        x = nn.Dense(features=10)(x)
        return x


model = LeNet5()


@jax.jit
def predict(params: dict, x: jax.Array) -> jax.Array:
    logits = model.apply({"params": params}, x)
    return jnp.argmax(logits, axis=-1)


@jax.jit
def accuracy(params: dict, x: jax.Array, y: jax.Array) -> jax.Array:
    return 100 * jnp.mean(predict(params, x) == y)


def make_loss_fn():
    # loss closes over model only; safe to jit
    @jax.jit
    def _loss(params, x, y):
        logits = model.apply({"params": params}, x)
        return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, y))

    return _loss


def make_train_step(optimizer, loss_fn):
    @jax.jit
    def _step(params, opt_state, x, y):
        loss_value, grads = jax.value_and_grad(loss_fn)(params, x, y)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value

    return _step


def train_one_run(seed: int, lr: float, wd: float, num_epochs: int, batch_size: int):
    """Run one full training: returns per-epoch train/test acc and the final values."""
    optimizer = optax.adamw(learning_rate=lr, weight_decay=wd)
    loss_fn = make_loss_fn()
    train_step = make_train_step(optimizer, loss_fn)

    key = random.PRNGKey(seed)
    params = model.init(key, jnp.ones([1, 32, 32, 1]))["params"]
    opt_state = optimizer.init(params)

    n = x_train_j.shape[0]
    train_curve, test_curve = [], []

    for epoch in range(num_epochs):
        # shuffle once per epoch
        perm = random.permutation(random.PRNGKey(seed * 1000 + epoch), n)
        for i in tqdm(
            range(0, n, batch_size),
            desc=f"seed={seed} lr={lr:.0e} wd={wd:.0e} epoch={epoch}",
            leave=False,
        ):
            idx = perm[i : i + batch_size]
            params, opt_state, _ = train_step(
                params, opt_state, x_train_j[idx], y_train_j[idx]
            )

        tr = float(accuracy(params, x_train_j, y_train_j))
        te = float(accuracy(params, x_test_j, y_test_j))
        train_curve.append(tr)
        test_curve.append(te)
        print(f"  epoch {epoch}: train = {tr:.2f}%  test = {te:.2f}%")

    return {
        "train_curve": train_curve,
        "test_curve": test_curve,
        "final_train": train_curve[-1],
        "final_test": test_curve[-1],
    }


# ------------------------------------------------------------------------------
# Experiment 1: 3 seeds at the "default" hyperparameters → headline + variance
# ------------------------------------------------------------------------------
SEEDS = (10, 42, 2024)
DEFAULT_LR = 1e-3
DEFAULT_WD = 1e-4
NUM_EPOCHS = 10
BATCH_SIZE = 128

print("\n" + "#" * 72)
print(f"# Experiment 1: 3-seed sweep at lr={DEFAULT_LR}, wd={DEFAULT_WD}")
print("#" * 72)

seed_results = {}
for seed in SEEDS:
    print(f"\n--- seed = {seed} ---")
    seed_results[seed] = train_one_run(seed, DEFAULT_LR, DEFAULT_WD, NUM_EPOCHS, BATCH_SIZE)


# ------------------------------------------------------------------------------
# Experiment 2: hyperparameter grid at one seed → LR / weight-decay sensitivity
# ------------------------------------------------------------------------------
GRID_SEED = 10
LR_GRID = (1e-2, 1e-3, 1e-4)
WD_GRID = (1e-3, 1e-4, 1e-5)

print("\n" + "#" * 72)
print(f"# Experiment 2: (lr, wd) grid at seed={GRID_SEED}")
print("#" * 72)

grid_results = {}
for lr in LR_GRID:
    for wd in WD_GRID:
        print(f"\n--- lr = {lr}, wd = {wd} ---")
        grid_results[(lr, wd)] = train_one_run(
            GRID_SEED, lr, wd, NUM_EPOCHS, BATCH_SIZE
        )


# ------------------------------------------------------------------------------
# Summary tables for easy reporting
# ------------------------------------------------------------------------------
print("\n" + "=" * 72)
print(f"SUMMARY 1: per-epoch test accuracy across {len(SEEDS)} seeds "
      f"(lr={DEFAULT_LR}, wd={DEFAULT_WD})")
print("=" * 72)
header = "Epoch | " + " | ".join(f"seed={s:<5}" for s in SEEDS) + " | mean"
print(header)
print("-" * len(header))
for e in range(NUM_EPOCHS):
    vals = [seed_results[s]["test_curve"][e] for s in SEEDS]
    mean_v = sum(vals) / len(vals)
    row = f"{e:>5} | " + " | ".join(f"{v:>9.2f}%" for v in vals) + f" | {mean_v:>5.2f}%"
    print(row)

print(f"\n{'Seed':<8} {'Train acc':>12} {'Test acc':>12}")
for s in SEEDS:
    print(f"{s:<8} {seed_results[s]['final_train']:>11.2f}% {seed_results[s]['final_test']:>11.2f}%")
final_tests = [seed_results[s]["final_test"] for s in SEEDS]
final_trains = [seed_results[s]["final_train"] for s in SEEDS]
mean_test = sum(final_tests) / len(final_tests)
mean_train = sum(final_trains) / len(final_trains)
print(f"\nMean final test  accuracy: {mean_test:.2f}%  (spread = {max(final_tests) - min(final_tests):.2f} pp)")
print(f"Mean final train accuracy: {mean_train:.2f}%  (spread = {max(final_trains) - min(final_trains):.2f} pp)")

print("\n" + "=" * 72)
print(f"SUMMARY 2: final test accuracy on (lr, wd) grid (seed={GRID_SEED})")
print("=" * 72)
print(f"{'lr \\ wd':>10} | " + " | ".join(f"wd={wd:<7.0e}" for wd in WD_GRID))
print("-" * (12 + len(WD_GRID) * 13))
for lr in LR_GRID:
    row_vals = " | ".join(
        f"{grid_results[(lr, wd)]['final_test']:>9.2f}%" for wd in WD_GRID
    )
    print(f"{lr:>10.0e} | {row_vals}")

print(f"\n{'lr':>8} {'wd':>8} {'final train':>13} {'final test':>13}")
for lr in LR_GRID:
    for wd in WD_GRID:
        r = grid_results[(lr, wd)]
        print(f"{lr:>8.0e} {wd:>8.0e} {r['final_train']:>12.2f}% {r['final_test']:>12.2f}%")
