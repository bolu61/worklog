# %%
import jax
from datasets.synthetic import interleaved_process_dataset
from tqdm import tqdm
from worklog.alpha import WorkLogAlpha

key = jax.random.key(0xCAFEB0BA)


dataset_key, training_key, evaluation_key = jax.random.split(key, 3)

# %%
ACTION_COUNT = 4

model = WorkLogAlpha(action_count=ACTION_COUNT)

# %%
SAMPLES_COUNT = 8
BATCH_SIZE = 1

dataset = interleaved_process_dataset(
    key=dataset_key,
    size=SAMPLES_COUNT,
    interleaving=4,
    states=4,
    alphabet=ACTION_COUNT,
    shape=1,
    length=64,
)

# %%
EMA_ALPHA = 0.2
SAVE_FILE = "worklog_alpha.pickle"
steps = model.fit(training_key, dataset, BATCH_SIZE)
loss_total = 0

for i, loss in (pbar := tqdm(enumerate(steps), total=SAMPLES_COUNT // BATCH_SIZE)):
    loss_total += loss
    pbar.set_description(f"loss={loss_total / (i + 1):4.4f}")

with open(SAVE_FILE, "wb") as f:
    model.dump(f)

# %%
print(model.evaluate(evaluation_key, dataset))