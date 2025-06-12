import cv2
import numpy as np
import imageio
import os
import jax
import tqdm
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from octo.model.octo_model import OctoModel

 
ACTION_DIM_LABELS = ['x', 'y', 'z', 'yaw', 'pitch', 'roll', 'grasp']
WINDOW_SIZE = 2


model = OctoModel.load_pretrained("hf://rail-berkeley/octo-small")
# create RLDS dataset builder
builder = tfds.builder_from_directory(builder_dir='gs://gresearch/robotics/bridge/0.1.0/')
ds = builder.as_dataset(split='train[:1]')
# sample episode + resize to 256x256 (default third-person cam resolution)
episode = next(iter(ds))
 
steps = list(episode['steps'])
images = [cv2.resize(np.array(step['observation']['image']), (256, 256)) for step in steps]
 
# extract goal image & language instruction
goal_image = images[-1]
language_instruction = steps[0]['observation']['natural_language_instruction'].numpy().decode()
 
# visualize episode
print(f'Instruction: {language_instruction}')
# media.show_video(images, fps=10)

media_dir = "media"
if not os.path.exists(media_dir):
    os.makedirs(media_dir)
# Save images as a GIF
gif_path = "media/episode.gif"
imageio.mimsave(gif_path, images, fps=10)
print(f"GIF saved to {gif_path}")
 
# create `task` dict
task = model.create_tasks(goals={"image_primary": goal_image[None]})   # for goal-conditioned
task = model.create_tasks(texts=[language_instruction]) 

# run inference loop, this model only uses single image observations for bridge
# collect predicted and true actions
pred_actions, true_actions = [], []
for step in tqdm.tqdm(range(0, len(images) - WINDOW_SIZE + 1)):
    input_images = np.stack(images[step : step + WINDOW_SIZE])[None]
    observation = {
        'image_primary': input_images,
        'timestep_pad_mask': np.array([[True, True]]),
    }
 
    # this returns *normalized* actions --> we need to unnormalize using the dataset statistics
    norm_actions = model.sample_actions(observation, task, rng=jax.random.PRNGKey(0))
    norm_actions = norm_actions[0]   # remove batch
    actions = (
        norm_actions * model.dataset_statistics["bridge_dataset"]['action']['std']
        + model.dataset_statistics["bridge_dataset"]['action']['mean']
    )
 
    pred_actions.append(actions)
    true_actions.append(np.concatenate(
        (
            steps[step+1]['action']['world_vector'],
            steps[step+1]['action']['rotation_delta'],
            np.array(steps[step+1]['action']['open_gripper']).astype(np.float32)[None]
        ), axis=-1
    ))

 
# build image strip to show above actions
img_strip = np.concatenate(np.array(images[::3]), axis=1)
 
# set up plt figure
figure_layout = [
    ['image'] * len(ACTION_DIM_LABELS),
    ACTION_DIM_LABELS
]
plt.rcParams.update({'font.size': 12})
fig, axs = plt.subplot_mosaic(figure_layout)
fig.set_size_inches([45, 10])
 
# plot actions
pred_actions = np.array(pred_actions).squeeze()
true_actions = np.array(true_actions).squeeze()
for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
  # actions have batch, horizon, dim, in this example we just take the first action for simplicity
  axs[action_label].plot(pred_actions[:, 0, action_dim], label='predicted action')
  axs[action_label].plot(true_actions[:, action_dim], label='ground truth')
  axs[action_label].set_title(action_label)
  axs[action_label].set_xlabel('Time in one episode')
 
axs['image'].imshow(img_strip)
axs['image'].set_xlabel('Time in one episode (subsampled)')
plt.legend()
plt.tight_layout()
plot_path = "media/actions_plot.png"
plt.savefig(plot_path)
print(f"Plot saved to {plot_path}")