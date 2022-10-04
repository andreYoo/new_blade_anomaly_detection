import json
import torch

from dataset.tmp_sample import MyData
from networks.main import build_autoencoder
from optim.ae_trainer import AETrainer


class BaseAE(object):
    """A class for the Deep SAD method.

    Attributes:
        eta: Deep SAD hyperparameter eta (must be 0 < eta).
        c: Hypersphere center c.
        net_name: A string indicating the name of the neural network to use.
        net: The neural network phi.
        trainer: DeepSADTrainer to train a Deep SAD model.
        optimizer_name: A string indicating the optimizer to use for training the Deep SAD network.
        ae_net: The autoencoder network corresponding to phi for network weights pretraining.
        ae_trainer: AETrainer to train an autoencoder in pretraining.
        ae_optimizer_name: A string indicating the optimizer to use for pretraining the autoencoder.
        results: A dictionary to save the results.
        ae_results: A dictionary to save the autoencoder results.
    """

    def __init__(self, writter=None):
        """Inits DeepSAD with hyperparameter eta."""


        self.trainer = None
        self.optimizer_name = None

        self.net = None  # autoencoder network for pretraining
        self.trainer = None
        self.optimizer_name = None

        self.results = {
            'train_time': None,
            'test_auc': None,
            'test_time': None,
            'test_scores': None,
        }


        self.writter = writter

    def set_network(self):
        """Builds the neural network phi."""
        self.net = build_autoencoder()


    def train(self, dataset, optimizer_name: str = 'adam', lr: float = 0.001,n_epochs: int = 100,
                 lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
                 n_jobs_dataloader: int = 0):
        """Pretrains the weights for the Deep SAD network phi via autoencoder."""


        # Train
        self.optimizer_name = optimizer_name
        self.trainer = AETrainer(optimizer_name, lr=lr, n_epochs=n_epochs, lr_milestones=lr_milestones,
                                    batch_size=batch_size, weight_decay=weight_decay,device=device,
                                    n_jobs_dataloader=n_jobs_dataloader
                                 )
        self.net = self.trainer.train(dataset, self.net, self.writter)


        # # Test
        # self.trainer.test(dataset, self.net)
        #
        # # Get test results
        # self.results['test_auc'] = self.trainer.test_auc
        # self.results['test_time'] = self.trainer.test_time
        # self.results['test_scores'] = self.trainer.test_scores


    def save_model(self, export_model):
        """Save Deep SAD model to export_model."""

        net_dict = self.net.state_dict()

        torch.save({'ae_net_dict': net_dict}, export_model)

    def load_model(self, model_path, load_ae=False, map_location='cpu'):
        """Load Deep SAD model from model_path."""

        model_dict = torch.load(model_path, map_location=map_location)

        self.net.load_state_dict(model_dict['ae_net_dict'])


    def save_results(self, export_json):
        """Save results dict to a JSON-file."""
        with open(export_json, 'w') as fp:
            json.dump(self.results, fp)

    def save_ae_results(self, export_json):
        """Save autoencoder results dict to a JSON-file."""
        with open(export_json, 'w') as fp:
            json.dump(self.results, fp)
