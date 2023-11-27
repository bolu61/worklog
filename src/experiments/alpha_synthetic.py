# %%
import jax
from datasets import interleaved_ergodic_process_dataset
from tqdm import tqdm
from worklog.alpha import WorkLogAlpha

key = jax.random.key(0xcafeb0ba)

model_key, dataset_key = jax.random.split(key, 2)

# %%
ACTIONS_COUNT = 4

model = WorkLogAlpha(
    key=model_key,
    num_actions=ACTIONS_COUNT
)

# %%
SAMPLES_COUNT = 1024
BATCH_SIZE = 1

dataset = interleaved_ergodic_process_dataset(
    key=dataset_key,
    size=SAMPLES_COUNT,
    interleaving=4,
    states=4,
    alphabet=ACTIONS_COUNT,
    shape=1,
    length=64
)

# %%
EMA_ALPHA = 0.2
steps = model.fit(dataset, BATCH_SIZE)

loss_ema = next(steps)

for loss in (pbar := tqdm(steps, total=SAMPLES_COUNT // BATCH_SIZE)):
    loss_ema = EMA_ALPHA * loss + (1 - EMA_ALPHA) * loss_ema
    pbar.set_description(f"loss={loss_ema:4.4f}")

# %%
