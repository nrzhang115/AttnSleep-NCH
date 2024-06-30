import numpy as np
import torch
from base import BaseTrainer
from utils import inf_loop, MetricTracker
import torch.nn as nn

selected_d = {"outs": [], "trg": []}
class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader, fold_id,
                 valid_data_loader=None, class_weights=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config, fold_id)
        self.config = config
        self.data_loader = data_loader
        self.len_epoch = len(self.data_loader)

        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = optimizer
        self.log_step = int(data_loader.batch_size) * 1  # reduce this if you want more logs

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])

        self.fold_id = fold_id
        self.selected = 0
        self.class_weights = class_weights
        ##########################################################
        # Calculate expected iterations per epoch
        self.expected_train_iterations = len(self.data_loader.dataset) // self.data_loader.batch_size
        if len(self.data_loader.dataset) % self.data_loader.batch_size != 0:
            self.expected_train_iterations += 1

        if self.do_validation:
            self.expected_valid_iterations = len(self.valid_data_loader.dataset) // self.valid_data_loader.batch_size
            if len(self.valid_data_loader.dataset) % self.valid_data_loader.batch_size != 0:
                self.expected_valid_iterations += 1

        print(f"Expected train iterations per epoch: {self.expected_train_iterations}")
        if self.do_validation:
            print(f"Expected validation iterations per epoch: {self.expected_valid_iterations}")
        ##################################################################################################

    def _train_epoch(self, epoch, total_epochs):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
               total_epochs: Integer, the total number of epoch
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        overall_outs = []
        overall_trgs = []
        
        actual_iterations = 0
        
        for batch_idx, (data, target) in enumerate(self.data_loader):
            actual_iterations += 1
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)

            loss = self.criterion(output, target, self.class_weights, self.device)

            loss.backward()
            self.optimizer.step()

            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f} '.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item(),
                ))

            if batch_idx == self.len_epoch:
                break
            
        print(f"Actual train iterations in epoch {epoch}: {actual_iterations}")
        log = self.train_metrics.result()

        if self.do_validation:
            val_log, outs, trgs = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})
            if val_log["accuracy"] > self.selected:
                self.selected = val_log["accuracy"]
                selected_d["outs"] = outs
                selected_d["trg"] = trgs
            if epoch == total_epochs:
                overall_outs.extend(selected_d["outs"])
                overall_trgs.extend(selected_d["trg"])

            # THIS part is to reduce the learning rate after 10 epochs to 1e-4
            if epoch == 10:
                for g in self.lr_scheduler.param_groups:
                    g['lr'] = 0.0001

        return log, overall_outs, overall_trgs

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        # Reset Validation metrics 
        self.valid_metrics.reset()
        #######################################################
        # Resetting the accumulators for each epoch
        all_preds = []
        all_targets = []
        actual_iterations = 0
        #######################################################
        with torch.no_grad():
            outs = np.array([])
            trgs = np.array([])
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                actual_iterations += 1
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target, self.class_weights, self.device)

                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))
                ##################################################   
                # Collect predictions and targets
                preds = torch.argmax(output, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                ##################################################
                preds_ = output.data.max(1, keepdim=True)[1].cpu()

                outs = np.append(outs, preds_.cpu().numpy())
                trgs = np.append(trgs, target.data.cpu().numpy())
            print(f"Actual validation iterations in epoch {epoch}: {actual_iterations}")
        ########################################################################
        # Log prediction distribution for validation
        print(f"After Validation Epoch {epoch}:")
        print(f"Validation Prediction Distribution: {np.bincount(all_preds)}")
        print(f"Validation Target Distribution: {np.bincount(all_targets)}")
        ########################################################################
        
        return self.valid_metrics.result(), outs, trgs

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
