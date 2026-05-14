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


key = random.PRNGKey(10)
model = MLP()
params = model.init(key, jnp.ones([1, 784]), train=True)[
    "params"
]  # jnp.ones is provided as dummy input to define the input shape to the model
optimizer = optax.adamw(learning_rate=1e-3, weight_decay=1e-4)

opt_state = optimizer.init(params)


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


untrained_accuracy = accuracy(params, jnp.array(x_test), jnp.array(y_test))
print("Untrained model accuracy:", untrained_accuracy)


BATCH_SIZE = 128
for epoch in range(50):
    for i in (pbar := tqdm(range(0, len(x_train), BATCH_SIZE), desc=f"Epoch {epoch}")):
        key, subkey = random.split(key)

        params, opt_state, train_loss = training_step(
            params,
            opt_state,
            x_train[i : i + BATCH_SIZE],
            y_train[i : i + BATCH_SIZE],
            subkey,
        )

        if i % (BATCH_SIZE * 5) == 0:
            pbar.set_postfix(
                {
                    "EPOCH": epoch,
                    "STEP": i,
                    "TRAIN LOSS": jnp.mean(
                        loss(params, x_train[:1000], y_train[:1000], subkey)
                    ),
                    "TEST ACCURACY": accuracy(
                        params, jnp.array(x_test), jnp.array(y_test)
                    ),
                }
            )

test_accuracy = accuracy(params, jnp.array(x_test), jnp.array(y_test))
train_accuracy = accuracy(params, jnp.array(x_train), jnp.array(y_train))
print("Test accuracy:", test_accuracy, "Train accuracy", train_accuracy)
