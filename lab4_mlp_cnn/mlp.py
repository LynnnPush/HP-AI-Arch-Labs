import os

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from jax import grad, random
from tensorflow.keras.datasets import mnist

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(jax.devices())

# loading and normalizing the data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize
x_train, x_test = x_train.reshape(-1, 784), x_test.reshape(-1, 784)  # Flatten


class MLP(nn.Module):
    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        return x


key = random.PRNGKey(10)
model = MLP()
params = model.init(key, jnp.ones([1, 784]))[
    "params"
]  # jnp.ones is provided as dummy input to define the input shape to the model


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
    grads = grad(loss)(params, x, y)
    updated_params = {
        layer: {
            param: params[layer][param] - lr * grads[layer][param]
            for param in params[layer]
        }
        for layer in params
    }
    return updated_params


untrained_accuracy = accuracy(params, jnp.array(x_test), jnp.array(y_test))
print("Untrained model accuracy:", untrained_accuracy)


for epoch in range(10):
    for i in range(len(x_train)):
        params = training_step(params, x_train[i], y_train[i])
        if i % 4000 == 0:
            print(
                "EPOCH",
                epoch,
                "STEP",
                i,
                "TRAIN LOSS",
                jnp.mean(loss(params, x_train[:1000], y_train[:1000])),
                "TEST ACCURACY",
                accuracy(params, jnp.array(x_test), jnp.array(y_test)),
            )

test_accuracy = accuracy(params, jnp.array(x_test), jnp.array(y_test))
train_accuracy = accuracy(params, jnp.array(x_train), jnp.array(y_train))
print("Test accuracy:", test_accuracy, "Train accuracy", train_accuracy)
