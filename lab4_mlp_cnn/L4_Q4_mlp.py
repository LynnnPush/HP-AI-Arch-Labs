import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from flax import linen as nn
from jax import grad, random
from tensorflow.keras.datasets import mnist

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(jax.devices())

# loading and normalizing the data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize
x_train, x_test = x_train.reshape(-1, 784), x_test.reshape(-1, 784)  # Flatten

# pre-convert to jax arrays so all batches stay on-device
x_train_j = jnp.array(x_train)
y_train_j = jnp.array(y_train)
x_test_j = jnp.array(x_test)
y_test_j = jnp.array(y_test)


class MLP(nn.Module):
    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        return x


model = MLP()


@jax.jit
def loss(params: dict, x: jax.Array, y: jax.Array) -> jax.Array:

    logits = model.apply(
        {"params": params}, x
    )  # forward pass of the model, calling the __call__ function and returning the output
    return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, y))


@jax.jit
def predict(params: dict, x: jax.Array) -> jax.Array:
    logits = model.apply({"params": params}, x)
    return jnp.argmax(logits, axis=-1)


@jax.jit
def accuracy(params: dict, x: jax.Array, y: jax.Array) -> jax.Array:
    predictions = predict(params, x)
    return 100 * jnp.mean(predictions == y)


@jax.jit
def training_step(
    params: dict[str, dict[str, jax.Array]],
    x: jax.Array,
    y: jax.Array,
    lr: float = 0.001,
) -> dict[str, dict[str, jax.Array]]:
    # mean-loss gradient over the batch, then plain SGD update
    grads = grad(loss)(params, x, y)
    updated_params = {
        layer: {
            param: params[layer][param] - lr * grads[layer][param]
            for param in params[layer]
        }
        for layer in params
    }
    return updated_params


# small-training-set experiment: 1000 samples, 20 epochs, batch size 64
LEARNING_RATE = 1.6e-3
BATCH_SIZE = 64
NUM_EPOCHS = 20
TRAIN_SUBSET = 1000

# restrict training data to the first TRAIN_SUBSET samples; test set unchanged
x_train_small = x_train_j[:TRAIN_SUBSET]
y_train_small = y_train_j[:TRAIN_SUBSET]
n_train = x_train_small.shape[0]

print("\n" + "=" * 70)
print(
    f"Training set = {n_train} samples, batch = {BATCH_SIZE}, "
    f"epochs = {NUM_EPOCHS}, lr = {LEARNING_RATE}"
)
print("=" * 70)

key = random.PRNGKey(10)
params = model.init(key, jnp.ones([1, 784]))["params"]
print("Untrained model accuracy:", accuracy(params, x_test_j, y_test_j))

# track both train and test accuracy per epoch to make over-/under-fitting visible
epoch_train_acc = []
epoch_test_acc = []
for epoch in range(NUM_EPOCHS):
    # fresh shuffle each epoch so batch composition varies
    perm = random.permutation(random.PRNGKey(epoch), n_train)
    for start in range(0, n_train, BATCH_SIZE):
        idx = perm[start : start + BATCH_SIZE]
        params = training_step(params, x_train_small[idx], y_train_small[idx], LEARNING_RATE)

    tr = float(accuracy(params, x_train_small, y_train_small))
    te = float(accuracy(params, x_test_j, y_test_j))
    epoch_train_acc.append(tr)
    epoch_test_acc.append(te)
    print(f"  epoch {epoch:>2}: train = {tr:.2f}%  test = {te:.2f}%  gap = {tr - te:+.2f}")

final_train = float(accuracy(params, x_train_small, y_train_small))
final_test = float(accuracy(params, x_test_j, y_test_j))

# summary table for easy reporting
print("\n" + "=" * 70)
print("SUMMARY: per-epoch train vs test accuracy (1000-sample training set)")
print("=" * 70)
print(f"{'Epoch':>5} | {'Train acc':>10} | {'Test acc':>10} | {'Gap':>8}")
print("-" * 45)
for epoch in range(NUM_EPOCHS):
    print(
        f"{epoch:>5} | {epoch_train_acc[epoch]:>9.2f}% | "
        f"{epoch_test_acc[epoch]:>9.2f}% | {epoch_train_acc[epoch] - epoch_test_acc[epoch]:>+7.2f}"
    )

print(f"\nFinal train accuracy: {final_train:.2f}%")
print(f"Final test  accuracy: {final_test:.2f}%")
print(f"Generalization gap  : {final_train - final_test:+.2f} percentage points")

# plot train/test accuracy curves to visualize the generalization gap
epochs = list(range(1, NUM_EPOCHS + 1))
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(epochs, epoch_train_acc, marker="o", label="Train accuracy")
ax.plot(epochs, epoch_test_acc, marker="s", label="Test accuracy")
# shaded band makes the overfitting gap visually obvious
ax.fill_between(epochs, epoch_test_acc, epoch_train_acc, alpha=0.15, label="Generalization gap")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy (%)")
ax.set_title(
    f"Train vs Test accuracy ({TRAIN_SUBSET} train samples, "
    f"bs={BATCH_SIZE}, lr={LEARNING_RATE})"
)
ax.set_xticks(epochs)
ax.grid(True, alpha=0.3)
ax.legend(loc="lower right")
fig.tight_layout()

out_path = os.path.join(os.path.dirname(__file__), "mlp_train_test_accuracy.png")
fig.savefig(out_path, dpi=120)
print(f"\nSaved accuracy curve plot to: {out_path}")
