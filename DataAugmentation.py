import numpy as np
import random


#Augmentation Functions
def add_noise(x, noise_std=0.01):
    return x + np.random.normal(0, noise_std, x.shape)

def horizontal_flip(x):
    x = x.copy()
    x[:, ::2] = 1.0 - x[:, ::2]  # Flip x-coordinates
    return x

def scale(x, min_scale=0.9, max_scale=1.1):
    center = np.mean(x, axis=1, keepdims=True)
    scale_factor = np.random.uniform(min_scale, max_scale)
    return (x - center) * scale_factor + center

def jitter(x, jitter_std=0.02):
    shift = np.random.normal(0, jitter_std, (1, x.shape[1]))
    return x + shift

def temporal_shift(x, max_shift=5):
    shift = np.random.randint(-max_shift, max_shift + 1)
    if shift > 0:
        return np.pad(x[:-shift], ((shift, 0), (0, 0)), mode='constant')
    elif shift < 0:
        return np.pad(x[-shift:], ((0, -shift), (0, 0)), mode='constant')
    else:
        return x


# Selecting random augmentation subsets
AUGMENTATIONS = [add_noise, horizontal_flip, scale, jitter, temporal_shift]

def apply_random_augmentations(x, k=2):
    ops = random.sample(AUGMENTATIONS, k)
    x_aug = x.copy()
    for op in ops:
        x_aug = op(x_aug)
    return x_aug


# Applying to dataset
def augment_dataset(x_path="X_60.npy", y_path="y_60.npy", save_prefix="X_60_augmented"):
    X = np.load(x_path)
    y = np.load(y_path)

    augmented_X = []
    augmented_y = []

    for i in range(len(X)):
        original = X[i]
        label = y[i]

        # Append original
        augmented_X.append(original)
        augmented_y.append(label)

        # Generate 3 different augmented versions
        for _ in range(3):
            aug = apply_random_augmentations(original, k=2)
            augmented_X.append(aug)
            augmented_y.append(label)

    X_aug = np.array(augmented_X)
    y_aug = np.array(augmented_y)

    np.save(f"{save_prefix}.npy", X_aug)
    np.save(f"{save_prefix.replace('X', 'y')}.npy", y_aug)

    print(f"✅ Augmented dataset saved with {X_aug.shape[0]} samples (original: {len(X)}, total: {len(X_aug)})")

# Usage
if __name__ == "__main__":
    augment_dataset()
