from datetime import datetime
from types import SimpleNamespace
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from utils import set_seed, generate_vit_paths, get_layers, Pruner, Quantizer, LRDer


class Trainer:
    def __init__(self, cfg: SimpleNamespace, model, loaders: Tuple, num_classes: int, augment_op=None):
        self.cfg = cfg
        self.model = model
        self.trainloader, self.valloader, self.testloader = loaders
        self.num_classes = num_classes
        self.augment_op = augment_op

        set_seed(self.cfg.seed)

        # Compression setup
        layers_str = generate_vit_paths()
        layers = get_layers(self.model, layers_str)
        self.pruner = Pruner(layers, self.cfg.prune_granularity, self.cfg.prune_ratio, self.cfg.prune_lambda_w) if self.cfg.prune_model else None
        self.quantizer = Quantizer(layers) if self.cfg.quantize_model else None
        self.lowranker = LRDer(self.model, layers_str, self.cfg.lrd_rank) if self.cfg.lrd_model else None

        # Device and optim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay)

        # logging
        self.writer = SummaryWriter(log_dir=getattr(self.cfg, 'tb_path', None))

        # schedules
        self.total_train_steps = self.cfg.total_epochs * len(self.trainloader)
        self.lr_warmup_steps = self.cfg.lr_warmup_epochs * len(self.trainloader)
        self.prune_vcon_steps = self.cfg.prune_vcon_epochs * len(self.trainloader) // self.cfg.batch_accumulation
        self.quantize_vcon_steps = self.cfg.quantize_vcon_epochs * len(self.trainloader) // self.cfg.batch_accumulation
        self.lrd_vcon_steps = self.cfg.lrd_vcon_epochs * len(self.trainloader) // self.cfg.batch_accumulation

        self.prune_beta = 1 if self.prune_vcon_steps > 0 else 0
        self.quantize_beta = 1 if self.quantize_vcon_steps > 0 else 0
        self.lrd_beta = 1 if self.lrd_vcon_steps > 0 else 0

        self.prune_vcon_steps_remaining = self.prune_vcon_steps
        self.quantize_vcon_steps_remaining = self.quantize_vcon_steps
        self.lrd_vcon_steps_remaining = self.lrd_vcon_steps

    def _maybe_augment(self, inputs, labels):
        if self.cfg.use_cutmix_mixup and self.augment_op is not None:
            return self.augment_op(inputs, labels)
        return inputs, labels

    def test(self):
        correct = 0
        total = 0
        self.model.eval()
        if self.pruner: self.pruner.prune_model(0.0)
        if self.quantizer: self.quantizer.quantize_model(0.0)
        with torch.no_grad():
            for images, labels in self.testloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        if self.quantizer: self.quantizer.restore_model()
        if self.pruner: self.pruner.restore_model()
        acc = 100.0 * correct / total if total else 0.0
        print(f'Accuracy of the network on the {len(self.testloader)*self.cfg.batch_size} test images: {acc}%')
        return acc

    def fit(self):
        if getattr(self.cfg, 'test', False):
            return self.test()

        start_time = datetime.now()
        last_time = start_time
        progressless_epochs = 0
        best_val_loss = float('inf')
        self.optimizer.zero_grad()

        for epoch in range(self.cfg.total_epochs):
            running_loss = 0.0
            running_output_loss = 0.0
            correct = 0
            total = 0
            if self.pruner: self.pruner.reset_SAD()

            self.model.train()
            for i, (inputs, labels) in enumerate(self.trainloader, 0):
                train_step = epoch * len(self.trainloader) + i

                # LR warmup
                if self.lr_warmup_steps > 0 and train_step <= self.lr_warmup_steps:
                    self.optimizer.param_groups[0]['lr'] = train_step * (self.cfg.learning_rate - self.cfg.lr_warmup_start) / max(1, self.lr_warmup_steps) + self.cfg.lr_warmup_start

                # LR cosine decay
                if train_step > self.cfg.lr_warmup_epochs and self.cfg.lr_cosine_decay:
                    self.optimizer.param_groups[0]['lr'] = self.cfg.lr_minimum + 0.5 * (self.cfg.learning_rate - self.cfg.lr_minimum) * (1 + np.cos(np.pi * train_step / max(1, self.total_train_steps)))

                inputs, labels = inputs.to(self.device), labels.to(self.device)
                inputs, labels = self._maybe_augment(inputs, labels)

                if self.pruner: self.pruner.prune_model(self.prune_beta)
                if self.quantizer: self.quantizer.quantize_model(self.quantize_beta)
                if self.lowranker: self.lowranker.upd_beta(self.lrd_beta)

                outputs = self.model(inputs)
                output_loss = self.criterion(outputs.logits, labels)
                loss = output_loss
                loss.backward()

                if self.quantizer: self.quantizer.restore_model()
                if self.pruner:
                    self.pruner.restore_model()
                    self.pruner.regularize_SAD()

                if (i + 1) % self.cfg.batch_accumulation == 0:
                    if self.cfg.clipping_global_norm:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    # update betas only when we step
                    if self.prune_vcon_steps_remaining > 0:
                        self.prune_vcon_steps_remaining -= 1
                        self.prune_beta -= 1 / max(1, self.prune_vcon_steps)
                    else:
                        self.prune_beta = 0
                    if self.quantize_vcon_steps_remaining > 0:
                        self.quantize_vcon_steps_remaining -= 1
                        self.quantize_beta -= 1 / max(1, self.quantize_vcon_steps)
                    else:
                        self.quantize_beta = 0
                    if self.lrd_vcon_steps_remaining > 0:
                        self.lrd_vcon_steps_remaining -= 1
                        self.lrd_beta -= 1 / max(1, self.lrd_vcon_steps)
                    else:
                        self.lrd_beta = 0

                # metrics
                running_loss += float(loss.item())
                running_output_loss += float(output_loss.item())
                total += labels.size(0)
                if self.cfg.use_cutmix_mixup:
                    _, maxlabels = torch.max(labels, 1)
                    correct += (torch.max(outputs.logits, 1)[1] == maxlabels).sum().item()
                else:
                    correct += (torch.max(outputs.logits, 1)[1] == labels).sum().item()

                if self.pruner: self.pruner.update_SAD()

                train_accuracy = (100 * correct / total) if total else 0.0
                train_output_loss = running_output_loss * self.cfg.batch_size / max(1, total)
                print(f'[{epoch+1}, {i+1:4d}] output_loss: {train_output_loss:1.3e} acc: {train_accuracy:3.2f}%', end='\r')

            print(f'[{epoch+1}] epoch complete.              ')

            # Validation
            running_loss = 0.0
            correct = 0
            total = 0
            self.model.eval()
            if self.pruner: self.pruner.prune_model(self.prune_beta)
            if self.quantizer: self.quantizer.quantize_model(self.quantize_beta)
            with torch.no_grad():
                for inputs, labels in self.valloader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs.logits, labels)
                    running_loss += float(loss.item())
                    total += labels.size(0)
                    correct += (torch.max(outputs.logits, 1)[1] == labels).sum().item()
            if self.quantizer: self.quantizer.restore_model()
            if self.pruner: self.pruner.restore_model()

            validation_accuracy = (100 * correct / total) if total else 0.0
            validation_loss = running_loss * self.cfg.batch_size / max(1, total)
            print(f'[val] loss: {validation_loss:1.3e} acc: {validation_accuracy:3.2f}%')

            # TB
            self.writer.add_scalar('Loss/validation', validation_loss, self.cfg.starting_epoch + epoch + 1)
            self.writer.add_scalar('Accuracy/validation', validation_accuracy, self.cfg.starting_epoch + epoch + 1)
            self.writer.add_scalar('Learning_rate', self.optimizer.param_groups[0]['lr'], self.cfg.starting_epoch + epoch + 1)
            if self.pruner:
                self.writer.add_scalar('SAD', self.pruner.get_SAD(), epoch + 1)

            # plateau schedule
            if epoch > self.cfg.lr_warmup_epochs:
                if validation_loss > best_val_loss:
                    progressless_epochs += 1
                else:
                    best_val_loss = validation_loss
                if progressless_epochs > self.cfg.lr_patience:
                    self.optimizer.param_groups[0]['lr'] *= self.cfg.lr_decay_factor
                    if self.optimizer.param_groups[0]['lr'] < self.cfg.lr_minimum:
                        self.optimizer.param_groups[0]['lr'] = self.cfg.lr_minimum
                    progressless_epochs = 0

            print(f'Epoch time: {datetime.now()-last_time}  Running time: {datetime.now()-start_time}')
            last_time = datetime.now()

            # save checkpoint
            torch.save(self.model.state_dict(), self.cfg.model_path)

        print('Training Complete')
        acc = self.test()
        return acc
