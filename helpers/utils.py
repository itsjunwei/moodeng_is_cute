import numpy as np
import torch
from torch.distributions.beta import Beta


def mixstyle(x, p=0.4, alpha=0.3, eps=1e-6):
    if np.random.rand() > p:
        return x
    batch_size = x.size(0)

    # changed from dim=[2,3] to dim=[1,3] - from channel-wise statistics to frequency-wise statistics
    f_mu = x.mean(dim=[1, 3], keepdim=True)
    f_var = x.var(dim=[1, 3], keepdim=True)

    f_sig = (f_var + eps).sqrt()  # compute instance standard deviation
    f_mu, f_sig = f_mu.detach(), f_sig.detach()  # block gradients
    x_normed = (x - f_mu) / f_sig  # normalize input
    lmda = Beta(alpha, alpha).sample((batch_size, 1, 1, 1)).to(x.device)  # sample instance-wise convex weights
    perm = torch.randperm(batch_size).to(x.device)  # generate shuffling indices
    f_mu_perm, f_sig_perm = f_mu[perm], f_sig[perm]  # shuffling
    mu_mix = f_mu * lmda + f_mu_perm * (1 - lmda)  # generate mixed mean
    sig_mix = f_sig * lmda + f_sig_perm * (1 - lmda)  # generate mixed standard deviation
    x = x_normed * sig_mix + mu_mix  # denormalize input using the mixed frequency statistics
    return x

class DataAugmentNumpyBase:
    """
    Base class for data augmentation for audio spectrogram of numpy array. This class does not alter label
    """
    def __init__(self, always_apply: bool = False, p: float = 0.5):
        self.always_apply = always_apply
        self.p = p

    def __call__(self, x: np.ndarray):
        if self.always_apply:
            return self.apply(x)
        else:
            if np.random.rand() < self.p:
                return self.apply(x)
            else:
                return x

    def apply(self, x: np.ndarray):
        raise NotImplementedError


class RandomCutoutNp(DataAugmentNumpyBase):
    """
    This data augmentation randomly cutout a rectangular area from the input image. Tested.
    """
    def __init__(self, always_apply: bool = False, p: float = 0.5, 
                 image_aspect_ratio: float = 1,
                 random_value: float = None):
        """
        :param always_apply: If True, always apply transform.
        :param p: If always_apply is false, p is the probability to apply transform.
        :param image_aspect_ratio: height/width ratio. For spectrogram: n_time_steps/ n_features.
        :param random_value: random value to fill in the cutout area. If None, randomly fill the cutout area with value
            between min and max of input.
        """
        super().__init__(always_apply, p)
        self.random_value = random_value

        # Params: s: area, r: height/width ratio.
        self.s_l = 0.02
        self.s_h = 0.3
        self.r_1 = 0.3
        self.r_2 = 1 / 0.3
        if image_aspect_ratio > 1:
            self.r_1 = self.r_1 * image_aspect_ratio
        elif image_aspect_ratio < 1:
            self.r_2 = self.r_2 * image_aspect_ratio

    def apply(self, x: np.ndarray) -> np.ndarray:
        """
        :param x: <(n_features, n_time_steps)>: input spectrogram.
        :return: random cutout x
        """

        assert x.ndim == 2, "Input Spectrogram is not 2-dimensional! Check again : {}".format(x.shape)
        x = x.T
        img_h = x.shape[-2]  # time frame dimension
        img_w = x.shape[-1]  # feature dimension
        min_value = np.min(x)
        max_value = np.max(x)

        # Initialize output
        output_img = x.copy()

        # random erase
        s = np.random.uniform(self.s_l, self.s_h) * img_h * img_w
        r = np.random.uniform(self.r_1, self.r_2)
        w = np.min((int(np.sqrt(s / r)), img_w - 1))
        h = np.min((int(np.sqrt(s * r)), img_h - 1))
        left = np.random.randint(0, img_w - w)
        top = np.random.randint(0, img_h - h)

        # Fill with random value
        if self.random_value is None:
            c = np.random.uniform(min_value, max_value)
        else:
            c = self.random_value
        output_img[top:top + h, left:left + w] = c

        return output_img.T


class SpecAugmentNp(DataAugmentNumpyBase):
    """
    This data augmentation randomly remove horizontal or vertical strips from image. Tested
    """
    def __init__(self, always_apply: bool = False, p: float = 0.5, time_max_width: int = None,
                 freq_max_width: int = None, n_time_stripes: int = 1, n_freq_stripes: int = 1):
        """
        :param always_apply: If True, always apply transform.
        :param p: If always_apply is false, p is the probability to apply transform.
        :param time_max_width: maximum time width to remove.
        :param freq_max_width: maximum freq width to remove.
        :param n_time_stripes: number of time stripes to remove.
        :param n_freq_stripes: number of freq stripes to remove.
        """
        super().__init__(always_apply, p)
        self.time_max_width = time_max_width
        self.freq_max_width = freq_max_width
        self.n_time_stripes = n_time_stripes
        self.n_freq_stripes = n_freq_stripes

    def apply(self, x: np.ndarray) -> np.ndarray:
        """
        :param x: <(n_features, n_timesteps)>: input spectrogram.
        :return: augmented spectrogram.
        """
        assert x.ndim == 2, 'Error: dimension of input spectrogram is not 2! Check again : {}'.format(x.shape)
        n_frames = x.shape[1]
        n_freqs = x.shape[0]
        min_value = np.min(x)
        max_value = np.max(x)
        if self.time_max_width is None:
            time_max_width = int(0.15 * n_frames)
        else:
            time_max_width = self.time_max_width
        time_max_width = np.max((1, time_max_width))
        if self.freq_max_width is None:
            freq_max_width = int(0.2 * n_freqs)
        else:
            freq_max_width = self.freq_max_width
        freq_max_width = np.max((1, freq_max_width))

        new_spec = x.copy()

        for i in np.arange(self.n_time_stripes):
            dur = np.random.randint(1, time_max_width, 1)[0]
            start_idx = np.random.randint(0, n_frames - dur, 1)[0]
            random_value = np.random.uniform(min_value, max_value, 1)
            new_spec[:, start_idx:start_idx + dur] = random_value

        for i in np.arange(self.n_freq_stripes):
            dur = np.random.randint(1, freq_max_width, 1)[0]
            start_idx = np.random.randint(0, n_freqs - dur, 1)[0]
            random_value = np.random.uniform(min_value, max_value, 1)
            new_spec[start_idx:start_idx + dur, :] = random_value

        return new_spec



class RandomCutoutHoleNp(DataAugmentNumpyBase):
    """
    This data augmentation randomly cutout a few small holes in the spectrogram. Tested.
    """
    def __init__(self, always_apply: bool = False, p: float = 0.5, n_max_holes: int = 8, max_h_size: int = 2,
                 max_w_size: int = 8, filled_value: float = None):
        """
        :param always_apply: If True, always apply transform.
        :param p: If always_apply is false, p is the probability to apply transform.
        :param n_max_holes: Maximum number of holes to cutout.
        :param max_h_size: Maximum time frames of the cutout holes.
        :param max_w_size: Maximum freq bands of the cutout holes.
        :param filled_value: random value to fill in the cutout area. If None, randomly fill the cutout area with value
            between min and max of input.
        """
        super().__init__(always_apply, p)
        self.n_max_holes = n_max_holes
        self.filled_value = filled_value

    def apply(self, x: np.ndarray):
        """
        :param x: <(n_features, n_time_steps)>: input spectrogram.
        :return: augmented spectrogram.
        """
        assert x.ndim == 2, 'Error: dimension of input spectrogram is not 2! Check RandomCutoutHoleNp : {}'.format(x.shape)
        img_h = x.shape[-1]  # time frame dimension
        img_w = x.shape[-2]  # feature dimension
        min_value = np.min(x)
        max_value = np.max(x)
        new_spec = x.copy()
        
        # Get a random number of cutout holes
        n_cutout_holes = np.random.randint(1, self.n_max_holes, 1)[0]

        self.max_w_size = int(0.04 * img_w)
        self.max_h_size = int(0.04 * img_h)

        for ihole in np.arange(n_cutout_holes):
            w = np.random.randint(1, self.max_w_size, 1)[0] # For frequency bins
            h = np.random.randint(1, self.max_h_size, 1)[0] # For time bins

            left = np.random.randint(0, img_w - w)
            top = np.random.randint(0, img_h - h)
    
            if self.filled_value is None:
                filled_value = np.random.uniform(min_value, max_value)
            else:
                filled_value = self.filled_value

            # Fill hole with the value
            new_spec[top:top + h, left:left + w] = filled_value

        return new_spec


class CompositeCutout(DataAugmentNumpyBase):
    """
    This data augmentation combine Random cutout, specaugment, cutout hole.
    """
    def __init__(self, always_apply: bool = False, p: float = 0.5, image_aspect_ratio: float = 1):
        """
        :param n_zero_channels: if given, these last n_zero_channels will be filled in zeros instead of random values
        :param is_filled_last_channels: if False, does not cutout n_zero_channels
        """
        super().__init__(always_apply, p)
        self.random_cutout = RandomCutoutNp(always_apply=True, image_aspect_ratio=image_aspect_ratio)
        self.spec_augment = SpecAugmentNp(always_apply=True)
        self.random_cutout_hole = RandomCutoutHoleNp(always_apply=True)

    def apply(self, x: np.ndarray):
        """
        For each sample in x, randomly choose one of the three augmentations to apply.
        
        :param x: np.ndarray of shape (Batch, Freq, Time)
        :return: Augmented np.ndarray.
        """
        output = x.copy()
        B = x.shape[0]

        for i in range(B):
            choice = np.random.randint(0, 3, 1)[0]
            sample = output[i]  # (shape: (F, T))
            if choice == 0:
                augmented = self.random_cutout.apply(sample)
            elif choice == 1:
                augmented = self.spec_augment.apply(sample)
            else:
                augmented = self.random_cutout_hole.apply(sample)
            output[i] = augmented
        return output


class RandomShiftUpDownNp(DataAugmentNumpyBase):
    """
    This data augmentation randomly shifts the spectrogram up or down.
    """
    def __init__(self, always_apply=True, p=0.5, freq_shift_range: int = None, direction: str = None, mode='reflect',
                 n_last_channels: int = 0):
        super().__init__(always_apply, p)
        self.freq_shift_range = freq_shift_range
        self.direction = direction
        self.mode = mode
        self.n_last_channels = n_last_channels

    def apply(self, x: np.ndarray):

        output = x.copy()
        B = x.shape[0]

        for i in range(B):
            sample = output[i]  # retain batch-dim (shape: (1, F, T))

            if np.random.rand() < self.p:
                pass
            else:
                n_features, n_timesteps = sample.shape
                if self.freq_shift_range is None:
                    self.freq_shift_range = int(n_features * 0.08)
                shift_len = np.random.randint(1, self.freq_shift_range, 1)[0]
                if self.direction is None:
                    direction = np.random.choice(['up', 'down'], 1)[0]
                else:
                    direction = self.direction
                new_spec = sample.copy()
                if direction == 'up':
                    new_spec = np.pad(new_spec, ((shift_len, 0), (0, 0)), mode=self.mode)[0:n_features, :]
                else:
                    new_spec = np.pad(new_spec, ((0, shift_len), (0, 0)), mode=self.mode)[shift_len:, :]

                output[i] = new_spec
        
        return output