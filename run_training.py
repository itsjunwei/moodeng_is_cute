import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import torch
import torchaudio
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import torch.nn.functional as F
import transformers
import wandb
import json
import math

from dataset.dcase24 import get_training_set, get_test_set, get_eval_set
from models.baseline import get_model, TestCNN
from helpers.utils import *
from helpers import nessi
from torchvision import transforms


class PLModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config  # results from argparse, contains all configurations for our experiment

        # module for resampling waveforms on the fly
        resample = torchaudio.transforms.Resample(
            orig_freq=self.config.orig_sample_rate,
            new_freq=self.config.sample_rate
        )

        # module to preprocess waveforms into log mel spectrograms
        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            win_length=config.window_length,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            f_min=config.f_min,
            f_max=config.f_max
        )

        self.mel = torch.nn.Sequential(
            resample,
            mel
        )
        
        self.augmentations = []
        if self.config.freqshift:
            self.augmentations.append(RandomShiftUpDownNp())
        if self.config.cutout:
            self.augmentations.append(CompositeCutout(image_aspect_ratio=(int(config.sample_rate / config.hop_length + 1)/config.n_mels)))
        print("Using Freqshift :{}\nUsing Cutout :{}".format(self.config.freqshift, self.config.cutout))

        self.mel_augment = transforms.Compose(self.augmentations)

        # the baseline model
        if config.model == "baseline":
            self.model = get_model(n_classes=config.n_classes,
                                in_channels=config.in_channels,
                                base_channels=config.base_channels,
                                channels_multiplier=config.channels_multiplier,
                                expansion_rate=config.expansion_rate
                                )
        else:
            self.model = TestCNN(num_classes=config.n_classes)

        self.device_ids = ['a', 'b', 'c', 's1', 's2', 's3', 's4', 's5', 's6']
        self.label_ids = ['airport', 'bus', 'metro', 'metro_station', 'park', 'public_square', 'shopping_mall',
                          'street_pedestrian', 'street_traffic', 'tram']
        # categorization of devices into 'real', 'seen' and 'unseen'
        self.device_groups = {'a': "real", 'b': "real", 'c': "real",
                              's1': "seen", 's2': "seen", 's3': "seen",
                              's4': "unseen", 's5': "unseen", 's6': "unseen"}

        # Device weights for balancing the skewed distribution of devices
        self.init_device_weights = [0.0883, 1.1821, 1.1821, 1.1821, 1.1821, 1.1821, 1.0, 1.0, 1.0]

        # If learnable, register as parameters; otherwise, register as fixed buffers.
        if config.learn_device:
            self.device_weight_logits = nn.Parameter(
                torch.tensor([math.log(w) for w in self.init_device_weights], dtype=torch.float)
            )
        else:
            self.register_buffer("device_weight_logits",
                                 torch.tensor([math.log(w) for w in self.init_device_weights], dtype=torch.float))
        print("Device weights: ", self.device_weight_logits)

        # Loss weights: if learnable, use a parameter; else, fix to equal weights.
        if config.learn_loss:
            self.loss_logits = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float))
        else:
            self.register_buffer("loss_logits", torch.tensor([0.5, 0.5], dtype=torch.float))
        print("Learnable Device Weights: {}\nLearnable Loss Weights:{}".format(self.config.learn_device, self.config.learn_loss))

        # pl 2 containers:
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def mel_forward(self, x):
        """
        :param x: batch of raw audio signals (waveforms)
        :return: log mel spectrogram
        """
        x = self.mel(x)
        if self.training:
            if len(self.augmentations):
                # Remove the channel dimension -> [B, 256, 65]
                x = x.squeeze(1) # Batch, Freq, Time
                # Convert tensor to numpy array
                x_np = x.detach().cpu().numpy()
                # Apply the augmentations
                x_aug = self.mel_augment(x_np)
                # Convert back to torch tensor and restore the channel dimension
                x = torch.from_numpy(x_aug).to(x.device).type_as(x).unsqueeze(1) # Batch, Channel (1), Freq, Time
        x = (x + 1e-5).log()
        if torch.isnan(x).any():
            raise ValueError("NaNs detected in log mel spectrogram")
        return x


    def forward(self, x, device_id, apply_mel=True):
        # Forward pass through the baseline model with device embeddings
        """
        :param x: batch of raw audio signals (waveforms)
        :param device_id: batch of device ids
        :return: final model predictions
        """
        if apply_mel:
            x = self.mel_forward(x) # Process the audio into log mel spectrograms
        x = self.model(x, device_id) # Input MelSpec + Device_ID into the model, get output
        return x   


    def configure_optimizers(self):
        """
        This is the way pytorch lightening requires optimizers and learning rate schedulers to be defined.
        The specified items are used automatically in the optimization loop (no need to call optimizer.step() yourself).
        :return: optimizer and learning rate scheduler
        """

        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )

        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1
        }
        return [optimizer], [lr_scheduler_config]
      

    def training_step(self, train_batch, batch_idx):
        """
        :param train_batch: contains one batch from train dataloader
        :param batch_idx
        :return: loss to update model parameters
        """
        x, files, labels, devices, cities = train_batch
        labels = labels.type(torch.LongTensor).to(self.device)
        devices = devices.to(torch.long)
        if devices.ndim > 1:
            devices = devices.squeeze()

        if self.config.mixstyle_p > 0:
            x = self.mel_forward(x)
            x = mixstyle(x, self.config.mixstyle_p, self.config.mixstyle_alpha)
            y_hat = self.forward(x, devices, apply_mel=False)  # Passing devices into forward method
        else:
            y_hat = self.forward(x, devices, apply_mel=True) # Did not convert to melspec because no mixstyle
        if torch.isnan(y_hat).any():
            raise ValueError("NaNs detected in model outputs")

        # Compute the standard average cross entropy loss
        avg_loss = F.cross_entropy(y_hat, labels, reduction="mean")
        # Compute the device-weighted loss (per-sample loss)
        losses = F.cross_entropy(y_hat, labels, reduction="none")

        # === Learnable Device Weights with Sum-to-1 Constraint ===
        # For seen devices (indices 0-5), use the device_weight_logits (learnable or fixed as per config)
        logits_seen = self.device_weight_logits[:6]  # shape: (6,)
        # For unseen devices (indices 6-8), fix logits to 0 (i.e. weight 1 before normalization)
        logits_unseen = torch.zeros(3, device=logits_seen.device, dtype=logits_seen.dtype)
        logits_full = torch.cat([logits_seen, logits_unseen], dim=0)  # shape: (9,)
        device_weights = torch.softmax(logits_full, dim=0) # Apply softmax to obtain normalized device weights (summing to 1)
        sample_weights = device_weights[devices]  # Now, for each sample, index the appropriate weight via its device index
        weighted_loss = losses * sample_weights
        # ============================================================

        # Combine losses: either use learnable loss weights or default to equal weighting.
        if self.config.learn_loss:
            weights = torch.softmax(self.loss_logits, dim=0)
            loss = weights[0] * avg_loss + weights[1] * weighted_loss.mean()
            self.log("loss_weight_avg", weights[0].detach().cpu(), prog_bar=True)
            self.log("loss_weight_device", weights[1].detach().cpu(), prog_bar=True)
        else:
            if self.config.only_avg:
                loss = avg_loss
            elif self.config.only_dev:
                loss = weighted_loss.mean()
            else:
                loss = 0.5 * avg_loss + 0.5 * weighted_loss.mean()

        self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'])
        self.log("epoch", self.current_epoch)
        self.log("train/loss", loss.detach().cpu(), prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        """
        The device embedding layer is defined as an nn.Embedding(9, embed_dim), 
        meaning there are nine rows (one for each possible device). 
        During training, only rows 0-5 get updated. 
        Without intervention, rows 6-8 remain at their initial random values. 
        By computing the mean of the updated embeddings for 0-5 
        and assigning that mean to rows 6-8 at the end of each epoch, 
        you ensure that these “missing” devices are given a representative embedding. 
        This regularizes the model for unseen devices and helps it generalize better 
        when these IDs occur during validation or testing.
        """
        # Update unseen device embeddings (6,7,8) to be the average of seen device embeddings (0-5)
        with torch.no_grad():
            # Get embeddings for seen device IDs (assumed to be 0-5)
            seen_indices = torch.tensor([0, 1, 2, 3, 4, 5], device=self.device)
            seen_embeddings = self.model.device_embedding(seen_indices)  # shape: [6, embed_dim]
            avg_seen = seen_embeddings.mean(dim=0, keepdim=True)  # shape: [1, embed_dim]
            
            # Update the embeddings for unseen device IDs
            for idx in [6, 7, 8]:
                self.model.device_embedding.weight.data[idx] = avg_seen.squeeze(0)


    def validation_step(self, val_batch, batch_idx):
        x, files, labels, devices, cities = val_batch
        devices = devices.to(torch.long)  # Ensure dtype is correct
        y_hat = self.forward(x, devices)
        labels = labels.type(torch.LongTensor).to(self.device)

        samples_loss = F.cross_entropy(y_hat, labels, reduction="none")

        # for computing accuracy
        _, preds = torch.max(y_hat, dim=1)
        n_correct_per_sample = (preds == labels)
        n_correct = n_correct_per_sample.sum()

        dev_names = [d.rsplit("-", 1)[1][:-4] for d in files]
        results = {'loss': samples_loss.mean(), "n_correct": n_correct,
                   "n_pred": torch.as_tensor(len(labels), device=self.device)}

        # log metric per device and scene
        for d in self.device_ids:
            results["devloss." + d] = torch.as_tensor(0., device=self.device)
            results["devcnt." + d] = torch.as_tensor(0., device=self.device)
            results["devn_correct." + d] = torch.as_tensor(0., device=self.device)
        for i, d in enumerate(dev_names):
            results["devloss." + d] = results["devloss." + d] + samples_loss[i]
            results["devn_correct." + d] = results["devn_correct." + d] + n_correct_per_sample[i]
            results["devcnt." + d] = results["devcnt." + d] + 1

        for l in self.label_ids:
            results["lblloss." + l] = torch.as_tensor(0., device=self.device)
            results["lblcnt." + l] = torch.as_tensor(0., device=self.device)
            results["lbln_correct." + l] = torch.as_tensor(0., device=self.device)
        for i, l in enumerate(labels):
            results["lblloss." + self.label_ids[l]] = results["lblloss." + self.label_ids[l]] + samples_loss[i]
            results["lbln_correct." + self.label_ids[l]] = \
                results["lbln_correct." + self.label_ids[l]] + n_correct_per_sample[i]
            results["lblcnt." + self.label_ids[l]] = results["lblcnt." + self.label_ids[l]] + 1
        results = {k: v.cpu() for k, v in results.items()}
        self.validation_step_outputs.append(results)

    def on_validation_epoch_end(self):
        # convert a list of dicts to a flattened dict
        outputs = {k: [] for k in self.validation_step_outputs[0]}
        for step_output in self.validation_step_outputs:
            for k in step_output:
                outputs[k].append(step_output[k])
        for k in outputs:
            outputs[k] = torch.stack(outputs[k])

        avg_loss = outputs['loss'].mean()
        acc = sum(outputs['n_correct']) * 1.0 / sum(outputs['n_pred'])

        logs = {'acc': acc, 'loss': avg_loss}

        # log metric per device and scene
        for d in self.device_ids:
            dev_loss = outputs["devloss." + d].sum()
            dev_cnt = outputs["devcnt." + d].sum()
            dev_corrct = outputs["devn_correct." + d].sum()
            logs["loss." + d] = dev_loss / dev_cnt
            logs["acc." + d] = dev_corrct / dev_cnt
            logs["cnt." + d] = dev_cnt
            # device groups
            logs["acc." + self.device_groups[d]] = logs.get("acc." + self.device_groups[d], 0.) + dev_corrct
            logs["count." + self.device_groups[d]] = logs.get("count." + self.device_groups[d], 0.) + dev_cnt
            logs["lloss." + self.device_groups[d]] = logs.get("lloss." + self.device_groups[d], 0.) + dev_loss

        for d in set(self.device_groups.values()):
            logs["acc." + d] = logs["acc." + d] / logs["count." + d]
            logs["lloss." + d] = logs["lloss." + d] / logs["count." + d]

        for l in self.label_ids:
            lbl_loss = outputs["lblloss." + l].sum()
            lbl_cnt = outputs["lblcnt." + l].sum()
            lbl_corrct = outputs["lbln_correct." + l].sum()
            logs["loss." + l] = lbl_loss / lbl_cnt
            logs["acc." + l] = lbl_corrct / lbl_cnt
            logs["cnt." + l] = lbl_cnt

        logs["macro_avg_acc"] = torch.mean(torch.stack([logs["acc." + l] for l in self.label_ids]))
        # prefix with 'val' for logging
        self.log_dict({"val/" + k: logs[k] for k in logs})
        self.validation_step_outputs.clear()

    def test_step(self, test_batch, batch_idx):
        x, files, labels, devices, cities = test_batch
        labels = labels.type(torch.LongTensor).to(self.device)
        devices = devices.to(torch.long)

        # maximum memory allowance for parameters: 128 KB
        # baseline has 61148 parameters -> we can afford 16-bit precision
        # since 61148 * 16 bit ~ 122 kB

        # assure fp16
        self.model.half()
        x = self.mel_forward(x)
        x = x.half()
        y_hat = self.forward(x, devices, apply_mel=False)
        samples_loss = F.cross_entropy(y_hat, labels, reduction="none")

        # for computing accuracy
        _, preds = torch.max(y_hat, dim=1)
        n_correct_per_sample = (preds == labels)
        n_correct = n_correct_per_sample.sum()

        dev_names = [d.rsplit("-", 1)[1][:-4] for d in files]
        results = {'loss': samples_loss.mean(), "n_correct": n_correct,
                   "n_pred": torch.as_tensor(len(labels), device=self.device)}

        # log metric per device and scene
        for d in self.device_ids:
            results["devloss." + d] = torch.as_tensor(0., device=self.device)
            results["devcnt." + d] = torch.as_tensor(0., device=self.device)
            results["devn_correct." + d] = torch.as_tensor(0., device=self.device)
        for i, d in enumerate(dev_names):
            results["devloss." + d] = results["devloss." + d] + samples_loss[i]
            results["devn_correct." + d] = results["devn_correct." + d] + n_correct_per_sample[i]
            results["devcnt." + d] = results["devcnt." + d] + 1

        for l in self.label_ids:
            results["lblloss." + l] = torch.as_tensor(0., device=self.device)
            results["lblcnt." + l] = torch.as_tensor(0., device=self.device)
            results["lbln_correct." + l] = torch.as_tensor(0., device=self.device)
        for i, l in enumerate(labels):
            results["lblloss." + self.label_ids[l]] = results["lblloss." + self.label_ids[l]] + samples_loss[i]
            results["lbln_correct." + self.label_ids[l]] = \
                results["lbln_correct." + self.label_ids[l]] + n_correct_per_sample[i]
            results["lblcnt." + self.label_ids[l]] = results["lblcnt." + self.label_ids[l]] + 1
        self.test_step_outputs.append(results)

    def on_test_epoch_end(self):
        print("Test epoch ended; computing metrics...")
        # convert a list of dicts to a flattened dict
        outputs = {k: [] for k in self.test_step_outputs[0]}
        for step_output in self.test_step_outputs:
            for k in step_output:
                outputs[k].append(step_output[k])
        for k in outputs:
            outputs[k] = torch.stack(outputs[k])

        avg_loss = outputs['loss'].mean()
        acc = sum(outputs['n_correct']) * 1.0 / sum(outputs['n_pred'])

        logs = {'acc': acc, 'loss': avg_loss}

        # log metric per device and scene
        for d in self.device_ids:
            dev_loss = outputs["devloss." + d].sum()
            dev_cnt = outputs["devcnt." + d].sum()
            dev_corrct = outputs["devn_correct." + d].sum()
            logs["loss." + d] = dev_loss / dev_cnt
            logs["acc." + d] = dev_corrct / dev_cnt
            logs["cnt." + d] = dev_cnt
            # device groups
            logs["acc." + self.device_groups[d]] = logs.get("acc." + self.device_groups[d], 0.) + dev_corrct
            logs["count." + self.device_groups[d]] = logs.get("count." + self.device_groups[d], 0.) + dev_cnt
            logs["lloss." + self.device_groups[d]] = logs.get("lloss." + self.device_groups[d], 0.) + dev_loss

        for d in set(self.device_groups.values()):
            logs["acc." + d] = logs["acc." + d] / logs["count." + d]
            logs["lloss." + d] = logs["lloss." + d] / logs["count." + d]

        for l in self.label_ids:
            lbl_loss = outputs["lblloss." + l].sum()
            lbl_cnt = outputs["lblcnt." + l].sum()
            lbl_corrct = outputs["lbln_correct." + l].sum()
            logs["loss." + l] = lbl_loss / lbl_cnt
            logs["acc." + l] = lbl_corrct / lbl_cnt
            logs["cnt." + l] = lbl_cnt

        logs["macro_avg_acc"] = torch.mean(torch.stack([logs["acc." + l] for l in self.label_ids]))
        # prefix with 'test' for logging
        self.log_dict({"test/" + k: logs[k] for k in logs})
        print("FINAL MACRO AVG ACC : {}".format(logs["macro_avg_acc"]))
        print("FINAL DEVICE WEIGHTS: ", self.device_weight_logits)
        self.test_step_outputs.clear()

    def predict_step(self, eval_batch, batch_idx, dataloader_idx=0):
        x, files, devices = eval_batch

        # assure fp16
        self.model.half()

        x = self.mel_forward(x)
        x = x.half()
        y_hat = self.forward(x, devices, apply_mel=False)

        return files, y_hat


def train(config):
    # logging is done using wandb
    wandb_logger = WandbLogger(
        project=config.project_name,
        notes="Baseline System for DCASE'24 Task 1.",
        tags=["DCASE24"],
        config=config,  # this logs all hyperparameters for us
        name=config.experiment_name
    )

    from pytorch_lightning.callbacks import ModelCheckpoint

    # Create a checkpoint callback to monitor validation accuracy
    checkpoint_callback = ModelCheckpoint(
        monitor="val/acc",  # Make sure your on_validation_epoch_end logs "val/acc"
        mode="max",         # "max" because higher accuracy is better
        save_top_k=1,       # Save only the best model
        verbose=True,
        filename="{epoch}-{val/acc:.3f}"  # Optional: customize the filename
    )


    # train dataloader
    assert config.subset in {100, 50, 25, 10, 5}, "Specify an integer value in: {100, 50, 25, 10, 5} to use one of " \
                                                  "the given subsets."
    roll_samples = config.orig_sample_rate * config.roll_sec
    train_dl = DataLoader(dataset=get_training_set(config.subset, roll=roll_samples),
                        #   worker_init_fn=worker_init_fn,
                          num_workers=config.num_workers,
                          batch_size=config.batch_size,
                          shuffle=True)

    test_dl = DataLoader(dataset=get_test_set(),
                        #  worker_init_fn=worker_init_fn,
                         num_workers=config.num_workers,
                         batch_size=config.batch_size)

    # create pytorch lightening module
    pl_module = PLModule(config)

    # get model complexity from nessi and log results to wandb
    sample = next(iter(test_dl))[0][0].unsqueeze(0)
    shape = pl_module.mel_forward(sample).size()
    macs, params = nessi.get_torch_size(pl_module.model, input_size=shape)
    # log MACs and number of parameters for our model
    wandb_logger.experiment.config['MACs'] = macs
    wandb_logger.experiment.config['Parameters'] = params

    # create the pytorch lightening trainer by specifying the number of epochs to train, the logger,
    # on which kind of device(s) to train and possible callbacks
    trainer = pl.Trainer(max_epochs=config.n_epochs,
                         check_val_every_n_epoch=1, # Run validation every n epochs
                         logger=wandb_logger,
                         accelerator='gpu',
                         devices=1,
                         precision=config.precision,
                         callbacks=[checkpoint_callback])
    # start training and validation for the specified number of epochs
    trainer.fit(pl_module, train_dl, test_dl)

    # final test step
    # here: use the validation split
    trainer.test(ckpt_path=checkpoint_callback.best_model_path, dataloaders=test_dl)

    wandb.finish()


def validate(config):

    import os
    from sklearn import preprocessing
    import pandas as pd
    import torch.nn.functional as F
    from dataset.dcase24 import dataset_config
    # logging is done using wandb
    wandb_logger = WandbLogger(
        project=config.project_name,
        notes="Baseline System for DCASE'24 Task 1.",
        tags=["DCASE24"],
        config=config,  # this logs all hyperparameters for us
        name=config.experiment_name
    )

    assert config.ckpt_id is not None, "A value for argument 'ckpt_id' must be provided."
    ckpt_dir = os.path.join(config.project_name, config.ckpt_id, "checkpoints")
    assert os.path.exists(ckpt_dir), f"No such folder: {ckpt_dir}"
    ckpt_file = os.path.join(ckpt_dir, "last.ckpt")
    assert os.path.exists(ckpt_file), f"No such file: {ckpt_file}. Implement your own mechanism to select" \
                                      f"the desired checkpoint."

    # train dataloader
    assert config.subset in {100, 50, 25, 10, 5}, "Specify an integer value in: {100, 50, 25, 10, 5} to use one of " \
                                                  "the given subsets."

    val_dl = DataLoader(dataset=get_test_set(),
                        #  worker_init_fn=worker_init_fn,
                         num_workers=config.num_workers,
                         batch_size=config.batch_size)

    # create pytorch lightening module
    pl_module = PLModule(config)
    pl_module = PLModule.load_from_checkpoint(ckpt_file, config=config)


    # create the pytorch lightening trainer by specifying the number of epochs to train, the logger,
    # on which kind of device(s) to train and possible callbacks
    trainer = pl.Trainer(max_epochs=config.n_epochs,
                         logger=wandb_logger,
                         accelerator='gpu',
                         devices=1,
                         precision=config.precision)

    trainer.test(pl_module, dataloaders=val_dl)


def evaluate(config):
    import os
    from sklearn import preprocessing
    import pandas as pd
    import torch.nn.functional as F
    from dataset.dcase24 import dataset_config

    assert config.ckpt_id is not None, "A value for argument 'ckpt_id' must be provided."
    ckpt_dir = os.path.join(config.project_name, config.ckpt_id, "checkpoints")
    assert os.path.exists(ckpt_dir), f"No such folder: {ckpt_dir}"
    ckpt_file = os.path.join(ckpt_dir, "last.ckpt")
    assert os.path.exists(ckpt_file), f"No such file: {ckpt_file}. Implement your own mechanism to select" \
                                      f"the desired checkpoint."

    # create folder to store predictions
    os.makedirs("predictions", exist_ok=True)
    out_dir = os.path.join("predictions", config.ckpt_id)
    os.makedirs(out_dir, exist_ok=True)

    # load lightning module from checkpoint
    pl_module = PLModule.load_from_checkpoint(ckpt_file, config=config)
    trainer = pl.Trainer(logger=False,
                         accelerator='gpu',
                         devices=1,
                         precision=config.precision)

    # evaluate lightning module on development-test split
    test_dl = DataLoader(dataset=get_test_set(),
                        #  worker_init_fn=worker_init_fn,
                         num_workers=config.num_workers,
                         batch_size=config.batch_size)

    # get model complexity from nessi
    sample = next(iter(test_dl))[0][0].unsqueeze(0).to(pl_module.device)
    shape = pl_module.mel_forward(sample).size()
    macs, params = nessi.get_torch_size(pl_module.model, input_size=shape)

    print(f"Model Complexity: MACs: {macs}, Params: {params}")
    assert macs <= nessi.MAX_MACS, "The model exceeds the MACs limit and must not be submitted to the challenge!"
    assert params <= nessi.MAX_PARAMS_MEMORY, \
        "The model exceeds the parameter limit and must not be submitted to the challenge!"

    allowed_precision = int(nessi.MAX_PARAMS_MEMORY / params * 8)
    print(f"ATTENTION: According to the number of model parameters and the memory limits that apply in the challenge,"
          f" you are allowed to use at max the following precision for model parameters: {allowed_precision} bit.")

    # obtain and store details on model for reporting in the technical report
    info = {}
    info['MACs'] = macs
    info['Params'] = params
    res = trainer.test(pl_module, test_dl)
    info['test'] = res

    # generate predictions on evaluation set
    eval_dl = DataLoader(dataset=get_eval_set(),
                        #  worker_init_fn=worker_init_fn,
                         num_workers=config.num_workers,
                         batch_size=config.batch_size)

    predictions = trainer.predict(pl_module, dataloaders=eval_dl)
    # all filenames
    all_files = [item[len("audio/"):] for files, _ in predictions for item in files]
    # all predictions
    all_predictions = torch.cat([torch.as_tensor(p) for _, p in predictions], 0)
    all_predictions = F.softmax(all_predictions, dim=1)

    # write eval set predictions to csv file
    df = pd.read_csv(dataset_config['meta_csv'], sep="\t")
    le = preprocessing.LabelEncoder()
    le.fit_transform(df[['scene_label']].values.reshape(-1))
    class_names = le.classes_
    df = {'filename': all_files}
    scene_labels = [class_names[i] for i in torch.argmax(all_predictions, dim=1)]
    df['scene_label'] = scene_labels
    for i, label in enumerate(class_names):
        df[label] = all_predictions[:, i]
    df = pd.DataFrame(df)

    # save eval set predictions, model state_dict and info to output folder
    df.to_csv(os.path.join(out_dir, 'output.csv'), sep='\t', index=False)
    torch.save(pl_module.model.state_dict(), os.path.join(out_dir, "model_state_dict.pt"))
    with open(os.path.join(out_dir, "info.json"), "w") as json_file:
        json.dump(info, json_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DCASE 24 argument parser')

    # general
    parser.add_argument('--project_name', type=str, default="DCASE24_Task1")
    parser.add_argument('--experiment_name', type=str, default="Baseline")
    parser.add_argument('--num_workers', type=int, default=0)  # number of workers for dataloaders
    parser.add_argument('--precision', type=str, default="32")

    # evaluation
    parser.add_argument('--evaluate', action='store_true')  # predictions on eval set
    parser.add_argument('--ckpt_id', type=str, default=None)  # for loading trained model, corresponds to wandb id

    # dataset
    # subset in {100, 50, 25, 10, 5}
    parser.add_argument('--orig_sample_rate', type=int, default=44100)
    parser.add_argument('--subset', type=int, default=5) # Lazy to keep adding into arguments

    # model
    parser.add_argument('--n_classes', type=int, default=10)  # classification model with 'n_classes' output neurons
    parser.add_argument('--in_channels', type=int, default=1)
    # adapt the complexity of the neural network (3 main dimensions to scale the baseline)
    parser.add_argument('--base_channels', type=int, default=32)
    parser.add_argument('--channels_multiplier', type=float, default=1.8)
    parser.add_argument('--expansion_rate', type=float, default=2.1)
    parser.add_argument('--model', type=str, default='baseline') # Determine the type of model to be used

    # training
    parser.add_argument('--n_epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--mixstyle_p', type=float, default=0.4)  # frequency mixstyle
    parser.add_argument('--mixstyle_alpha', type=float, default=0.3)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--roll_sec', type=int, default=0.1)  # roll waveform over time
    parser.add_argument('--freqshift', action='store_true')  # Apply Freqshift
    parser.add_argument('--cutout', action='store_true')  # Apply Cutout

    # peak learning rate (in cosinge schedule)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--warmup_steps', type=int, default=2000)
    parser.add_argument('--only_avg', action='store_true')  # Only Averaged Cross Entropy
    parser.add_argument('--only_dev', action='store_true')  # Only Device-weighted Cross Entropy

    # preprocessing
    parser.add_argument('--sample_rate', type=int, default=32000)
    parser.add_argument('--window_length', type=int, default=3072)  # in samples (corresponds to 96 ms)
    parser.add_argument('--hop_length', type=int, default=500)  # in samples (corresponds to ~16 ms)
    parser.add_argument('--n_fft', type=int, default=4096)  # length (points) of fft, e.g. 4096 point FFT
    parser.add_argument('--n_mels', type=int, default=256)  # number of mel bins
    parser.add_argument('--freqm', type=int, default=48)  # mask up to 'freqm' spectrogram bins
    parser.add_argument('--timem', type=int, default=0)  # mask up to 'timem' spectrogram frames
    parser.add_argument('--f_min', type=int, default=0)  # mel bins are created for freqs. between 'f_min' and 'f_max'
    parser.add_argument('--f_max', type=int, default=None)

    # New flags to enable learnable parameters
    parser.add_argument('--learn_device', action='store_true',
                        help="Make device weights learnable (otherwise, they remain fixed).")
    parser.add_argument('--learn_loss', action='store_true',
                        help="Make loss weights learnable (otherwise, they are fixed at equal weights).")

    args = parser.parse_args()
    # validate(args)
    if args.evaluate:
        evaluate(args)
    else:
        train(args)
