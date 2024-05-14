
# %%
import sys
from functools import partial

import jax
import jax.numpy as jnp
from cliffs_delta import cliffs_delta
from experiments.datasets.synthetic import masked_process_dataset
from scipy.stats import mannwhitneyu
from tqdm import tqdm
from worklog.alpha import WorkLogAlpha
from worklog.beta import WorkLogBeta

key = jax.random.key(0xCAFEB0BA)

dataset_key, training_key, evaluation_key = jax.random.split(key, 3)

SAMPLE_SIZE = 10_000
BATCH_SIZE = 10
EPOCH_COUNT = 8
TRAIN_RATIO = 0.9

dka, dkb, dkc, dkd = jax.random.split(dataset_key, 4)
datasets = {
    "easy": (
        masked_process_dataset(
            key=key,
            size=SAMPLE_SIZE,
            interleaving=4,
            states=4,
            alphabet=4,
            shape=1,
            length=10,
        ),
        4,
    ),
}

models = {
    4: lambda action_count: WorkLogBeta(4, 4, action_count, lr=1e-1),
    8: lambda action_count: WorkLogBeta(4, 8, action_count, lr=1),
    16: lambda action_count: WorkLogBeta(4, 16, action_count, lr=1),
    32: lambda action_count: WorkLogBeta(4, 32, action_count, lr=1),
}


@partial(jax.jit, static_argnums=(0,))
def onehot(a, o):
    return jnp.zeros(a, dtype=jnp.uint32).at[o].set(1)


@partial(jax.jit, static_argnums=(0,))
def throughput(a, os):
    return jnp.sum(jax.vmap(onehot, in_axes=(None, 0))(a, os), axis=0)


def run(name, model, dataset, action_count):
    trainset, evalset = dataset.split(int(len(dataset) * TRAIN_RATIO))
    for epoch in range(EPOCH_COUNT):
        total_loss = 0
        for i, loss in (
            pbar := tqdm(
                enumerate(model.fit(training_key, trainset, BATCH_SIZE)),
                total=len(trainset) // BATCH_SIZE,
            )
        ):
            total_loss += loss
            mean_loss = total_loss / (i + 1)
            pbar.set_description(f"training {name} epoch={epoch} loss={mean_loss:4.4f}")

    t_true = []
    t_pred = []

    loss = 0
    for i, (os_true, ek) in (
        pbar := tqdm(
            enumerate(zip(evalset, jax.random.split(key, len(evalset)))),
            total=len(evalset),
        )
    ):
        os_pred = model.sequence(ek, len(os_true))
        t_true += [throughput(action_count, os_true)]
        t_pred += [throughput(action_count, os_pred)]
        loss += -model.forward(os_true)
        pbar.set_description(f"evaluation {name} loss={loss / (i + 1):4.4f}")

    t_true = jnp.stack(t_true)
    t_pred = jnp.stack(t_pred)
    _, pvalue = mannwhitneyu(t_true, t_pred, axis=0)

    return {
        "delta": [
            cliffs_delta(t_true[:, i], t_pred[:, i])[0] for i in range(action_count)
        ],
        "perplexity": jnp.exp(loss / len(dataset)),
    }


# %%
result = {}
for model_name, model_fn in models.items():
    for dataset_name, (dataset, action_count) in datasets.items():
        name = (model_name, dataset_name)
        model = model_fn(action_count)
        result[name] = run(name, model, dataset, action_count)
        print(
            f"result {name} {result[name]}",
            file=sys.stderr,
            flush=True,
        )

print(result)

# %%
