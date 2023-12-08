# %%
import jax
import jax.numpy as jnp
from datasets.synthetic import masked_process_dataset
from tqdm import tqdm
from worklog.alpha import WorkLogAlpha

key = jax.random.key(0xCAFEB0BA)

dataset_key, training_key, evaluation_key = jax.random.split(key, 3)

# %%
CLUSTER_COUNT = 4
SEQUENCE_LENGTH = 4
ACTION_COUNT = 4

model = WorkLogAlpha(
    cluster_count=CLUSTER_COUNT,
    sequence_length=SEQUENCE_LENGTH,
    action_count=ACTION_COUNT,
)

# %%
TRAIN_SAMPLES_COUNT = 1024
EVAL_SAMPLES_COUNT = 256

dataset = masked_process_dataset(
    key=dataset_key,
    size=TRAIN_SAMPLES_COUNT + EVAL_SAMPLES_COUNT,
    interleaving=CLUSTER_COUNT,
    states=SEQUENCE_LENGTH,
    alphabet=ACTION_COUNT,
    shape=1,
    length=CLUSTER_COUNT * SEQUENCE_LENGTH,
)

trainset, evalset = dataset.split(TRAIN_SAMPLES_COUNT)

# %%
BATCH_SIZE = 1
steps = model.fit(training_key, dataset, BATCH_SIZE)

loss_total = 0
for i, loss in (pbar := tqdm(steps, total=len(trainset) // BATCH_SIZE)):
    loss_total += loss
    perplexity = jnp.exp(loss / i)
    pbar.set_description(f"loss={loss_total / (i + 1):4.4f}")

# %%
