import numpy as np
import tqdm
import os
import cv2

N_TRAIN_EPISODES = 12
N_VAL_EPISODES = 3

DATA_PATH = "/data/pulkitag/misc/marcel/simassets/PolicyLearning/demos/cupntrashrealimgteleop/"
TASK_DESCRIPTION = 'pick bowl and place in sink'
def create_fake_episode(idx, path):
    episode = []
    actions = np.load(DATA_PATH+f"actions_0_{idx}.npy")
    images = np.load(DATA_PATH+f"images_0_{idx}.npz")["arr_0"][0,:,0]
    states = np.load(DATA_PATH+f"states_0_{idx}.npy")
    for step in range(len(actions)):
        episode.append({
            'image': cv2.resize(images[step].transpose((1,2,0)), (256, 256), interpolation = cv2.INTER_LINEAR).astype(np.uint8),
            'state': states[step],
            'action': actions[step],
            'language_instruction': TASK_DESCRIPTION,
        })
    np.save(path, episode)


# create fake episodes for train and validation
print("Generating train examples...")
os.makedirs('data/train', exist_ok=True)
for i in tqdm.tqdm(range(N_TRAIN_EPISODES)):
    create_fake_episode(i, f'data/train/episode_{i}.npy')

print("Generating val examples...")
os.makedirs('data/val', exist_ok=True)
for i in tqdm.tqdm(range(N_VAL_EPISODES)):
    create_fake_episode(i, f'data/val/episode_{i}.npy')

print('Successfully created example data!')
