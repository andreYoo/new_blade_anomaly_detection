import click
import torch
import logging
import random
import numpy as np

from utils.config import Config
from AE import BaseAE
from dataset.main import load_dataset
from tensorboardX import SummaryWriter
import time
import os

###############################################################################
# Settings
################################################################################
@click.command()
@click.argument('xp_path', default='log', type=click.Path(exists=True))
@click.argument('data_path',  default='tmp_train_samples.h5', type=click.Path(exists=True))
@click.option('--load_config', type=click.Path(exists=True), default=None,help='Config JSON-file path (default: None).')
@click.option('--load_model', type=click.Path(exists=True), default=None, help='Model file path (default: None).')
@click.option('--device', type=str, default='cuda', help='Computation device to use ("cpu", "cuda", "cuda:2", etc.).')
@click.option('--optimizer_name', type=click.Choice(['adam']), default='adam',
              help='Name of the optimizer to use for autoencoder pretraining.')
@click.option('--lr', type=float, default=0.001,
              help='Initial learning rate for autoencoder pretraining. Default=0.001')
@click.option('--n_epochs', type=int, default=100, help='Number of epochs to train autoencoder.')
@click.option('--lr_milestone', default=[20,50,70,90], multiple=True,
              help='Lr scheduler milestones at which lr is multiplied by 0.1. Can be multiple and must be increasing.')
@click.option('--batch_size', type=int, default=6, help='Batch size for mini-batch autoencoder training.')
@click.option('--weight_decay', type=float, default=1e-6,
              help='Weight decay (L2 penalty) hyperparameter for autoencoder objective.')
@click.option('--n_jobs_dataloader', type=int, default=0,
              help='Number of workers for data loading. 0 means that the data will be loaded in the main process.')
@click.option('--monitor', type=bool, default=True,
              help='Moniting learnig process using Tensorboard')

def main(xp_path,data_path, load_config, load_model,device, optimizer_name, lr, n_epochs, lr_milestone, batch_size, weight_decay,
         n_jobs_dataloader,monitor):
    """
    Deep SAD, a method for deep semi-supervised anomaly detection.

    :arg DATASET_NAME: Name of the dataset to load.
    :arg NET_NAME: Name of the neural network to use.
    :arg XP_PATH: Export path for logging the experiment.
    :arg DATA_PATH: Root path of data.
    """

    # Get configuration
    cfg = Config(locals().copy())

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = xp_path + './log.txt'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


    if monitor==True:
        _time_rightnow = time.strftime('/%y%m%d_%H_%M_%S')
        log_file = xp_path + _time_rightnow
        if not os.path.exists(log_file):
            os.makedirs(log_file)
            logger.info('NO Dir for learning monitoring - It is generated in %s' % log_file)
            writter = SummaryWriter(log_file)


    # Print experimental setup
    logger.info('[Experiment detils]-------------------------------------------------------------------------')
    logger.info('Log file is %s' % log_file)
    logger.info('Data path is %s' % data_path)
    logger.info('Export path is %s' % xp_path)

    # If specified, load experiment config from JSON-file
    if load_config:
        cfg.load_config(import_json=load_config)
        logger.info('Loaded configuration from %s.' % load_config)


    # Default device to 'cpu' if cuda is not available
    if not torch.cuda.is_available():
        device = 'cpu'
    # Set the number of threads used for parallelizing CPU operations
    logger.info('Computation device: %s' % device)
    logger.info('Number of dataloader workers: %d' % n_jobs_dataloader)

    # Load data
    dataset = load_dataset(data_path,batch_size=batch_size,shuffle_train=True,num_workers=n_jobs_dataloader)
    # Log random sample of known anomaly classes if more than 1 class

    AE = BaseAE(writter)
    AE.set_network()


    # If specified, load Deep SAD model (center c, network weights, and possibly autoencoder weights)
    if load_model:
        AE.load_model(model_path=load_model, load_ae=True, map_location=device)
        logger.info('Loading model from %s.' % load_model)


    # Log pretraining details
    logger.info('[Training Parameter setting]-----------------------------------------------------------------')
    logger.info('Training optimizer: %s' % cfg.settings['optimizer_name'])
    logger.info('Training learning rate: %g' % cfg.settings['lr'])
    logger.info('Training epochs: %d' % cfg.settings['n_epochs'])
    logger.info('Training learning rate scheduler milestones: %s' % (cfg.settings['lr_milestone'],))
    logger.info('Training batch size: %d' % cfg.settings['batch_size'])
    logger.info('Training weight decay: %g' % cfg.settings['weight_decay'])

    # Pretrain model on dataset (via autoencoder)
    AE.train(dataset,optimizer_name=cfg.settings['optimizer_name'],
                     lr=cfg.settings['lr'],
                     n_epochs=cfg.settings['n_epochs'],
                     lr_milestones=cfg.settings['lr_milestone'],
                     batch_size=cfg.settings['batch_size'],
                     weight_decay=cfg.settings['weight_decay'],
                     device=device,
                     n_jobs_dataloader=n_jobs_dataloader)


    # Save results, model, and configuration
    AE.save_results(export_json=xp_path + '/results.json')
    AE.save_model(export_model=xp_path + '/model.tar')
    cfg.save_config(export_json=xp_path + '/config.json')



if __name__ == '__main__':
    main()
