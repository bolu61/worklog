# %%
import jax
import jax.numpy as jnp
from datasets.synthetic import masked_process_dataset
from tqdm import tqdm
from worklog.alpha import WorkLogAlpha

key = jax.random.key(0xCAFEB0BA)

dataset_key, training_key, evaluation_key = jax.random.split(key, 3)

CLUSTER_COUNT = 4
SEQUENCE_LENGTH = 4
ACTION_COUNT = 4
TRAIN_SAMPLES_COUNT = 1024
EVAL_SAMPLES_COUNT = 256
BATCH_SIZE = 1

# %%
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
model = WorkLogAlpha(
    cluster_count=CLUSTER_COUNT,
    sequence_length=SEQUENCE_LENGTH,
    action_count=ACTION_COUNT,
)

steps = model.fit(training_key, trainset, BATCH_SIZE)

total_loss = 0
for i, loss in (pbar := tqdm(steps, total=len(trainset) // BATCH_SIZE)):
    total_loss += loss
    mean_loss = total_loss / i
    perplexity = jnp.exp(total_loss / i)
    pbar.set_description(f"loss={mean_loss:4.4f} perplexity={perplexity:4.4f}")

# %%
