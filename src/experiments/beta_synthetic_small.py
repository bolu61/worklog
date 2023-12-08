# %%
import jax
from datasets.synthetic import masked_process_dataset
from tqdm import tqdm
from worklog.beta import WorkLogBeta

key = jax.random.key(0xCAFEB0BA)

dataset_key, training_key = jax.random.split(key, 2)

# %%
ACTION_COUNT = 4
SEQUENCE_LENGTH = 4
CLUSTER_COUNT = 4

# account for extra cluster 0 to catch missing classes
model = WorkLogBeta(
    cluster_count=CLUSTER_COUNT,
    sequence_length=SEQUENCE_LENGTH,
    action_count=ACTION_COUNT,
)

# %%
SAMPLES_COUNT = 1024
BATCH_SIZE = 1

dataset = masked_process_dataset(
    key=dataset_key,
    size=SAMPLES_COUNT * 8,
    interleaving=CLUSTER_COUNT,
    states=4,
    alphabet=ACTION_COUNT,
    shape=1,
    length=64,
)

trainset, evalset = dataset.split(SAMPLES_COUNT)

# %%
SAVE_FILE = "worklog_beta_small.pickle"
steps = model.fit(training_key, trainset, BATCH_SIZE)
loss_total = 0

for i, loss in (pbar := tqdm(enumerate(steps), total=SAMPLES_COUNT // BATCH_SIZE)):
    loss_total += loss
    pbar.set_description(f"loss={loss_total / (i + 1):4.4f}")

with open(SAVE_FILE, "wb") as f:
    model.dump(f)

# %%
