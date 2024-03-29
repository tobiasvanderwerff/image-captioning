import random
import re
import itertools
import logging
import argparse
import datetime
import gc
from pathlib import Path
from functools import partial

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from data.flickr import download_flickr8k
from data.data import Lang, FlickrDataset
from src.caption_utils import preprocess_tokens
from src.trainer import Trainer, TrainerConfig
from src.utils import set_seed, train_collate_fn, eval_collate_fn, make_predictions
from src.metrics import calculate_bleu_score
from src.models import EncoderDecoder, LSTMDecoder, ResNet50Encoder, ResNet34Encoder, VGGNetEncoder, MobileNetEncoder, DenseNetEncoder


def main(args):
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # Make deterministic
    set_seed(47)    

    # Enable cuDNN autotuner. This runs a short benchmark and selects the convolution algorithm
    # with the best performance.
    torch.backends.cudnn.benchmark = True  

    # Normalization data for pretrained PyTorch model
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # We resize the images and center crop them, as in the 'Show, attend and tell' paper. In the paper they
    # resize the shortest side to 256 while preserving the aspect ratio, and then apply a center crop to 
    # end up with an image of size 224x224. Note that for quicker experimentation, we can use a smaller image 
    # size (e.g. 128x128) to speed up training. 
    trnsf = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
#             transforms.Resize(128),
#             transforms.CenterCrop(128),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'eval': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
#             transforms.Resize(128),
#             transforms.CenterCrop(128),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    }

    # Download dataset
    root_path = Path(args.data_path)
    img_dir, ann_file = download_flickr8k(root_path)

    # Preprocess the captions
    img_captions_enc, tokens, word2idx, idx2word = preprocess_tokens(ann_file)
    lang = Lang(tokens, word2idx, idx2word)
    vocab_size = len(tokens)

    # Split data up into train, evaluation and test set. We use the predefined train/eval splits of Flickr8k
    ds_train = FlickrDataset(img_dir, img_captions_enc, lang, ann_file,
                             root_path/'Flickr_8k.trainImages.txt', 'train', trnsf=trnsf['train'])
    ds_eval = FlickrDataset(img_dir, img_captions_enc, lang, ann_file,
                            root_path/'Flickr_8k.devImages.txt', 'eval', trnsf=trnsf['eval'])
    ds_test = FlickrDataset(img_dir, img_captions_enc, lang, ann_file,
                            root_path/'Flickr_8k.testImages.txt', 'test', trnsf=trnsf['eval'])

    # Load hyperparameters from a configuration file
    f = open('hyperparams.yaml', 'r')
    config_data = yaml.load_all(f, Loader=yaml.Loader)

    for experiment in config_data:  # run the experiments

        save_folder = experiment['path_to_save']
        params = experiment['parameters']

        num_hidden = params['num_hidden']
        embedding_dim = params['embedding_dim']
        batch_size = params['batch_size']
        num_layers = params['num_layers']
        epochs = params['epochs']
        lr = params['lr']
        optimizer_name = params['optimizer']
        encoder_name = params['encoder']
    
        assert optimizer_name.lower() in ['adam', 'rmsprop', 'sgd']
        assert encoder_name.lower() in ['resnet34', 'resnet50', 'vggnet', 'mobilenet', 'densenet']
        
        # Set up paths for storing data
        save_folder = Path(save_folder)
        cp_path = root_path / save_folder / 'checkpoints'
        (root_path / save_folder).mkdir(exist_ok=True, parents=True)
        cp_path.mkdir(exist_ok=True)

        # Set up logging
        logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            datefmt='%d/%m/%Y %H:%M:%S',
                            level=logging.INFO,
                            handlers=[
                                logging.FileHandler(root_path / save_folder / 'train.log'),
                                logging.StreamHandler()
                            ])
        logger = logging.getLogger(__name__)

        logger.info(f'Saving logs in {root_path / save_folder}/train.log')
        logger.info(f'Saving checkpoints in {cp_path}')

        if encoder_name.lower() == 'resnet34':
            encoder = ResNet34Encoder(num_hidden)
        if encoder_name.lower() == 'resnet50':
            encoder = ResNet50Encoder(num_hidden)
        elif encoder_name.lower() == 'vggnet':
            encoder = VGGNetEncoder(num_hidden)
        elif encoder_name.lower() == 'mobilenet':
            encoder = MobileNetEncoder(num_hidden)
        elif encoder_name.lower() == 'densenet':
            encoder = DenseNetEncoder(num_hidden)

        decoder = LSTMDecoder(num_hidden, embedding_dim, vocab_size, encoder.annotation_dim,
                              lang.word2idx['<START>'], device, num_layers=num_layers)

        model = EncoderDecoder(encoder, decoder, device)
        model = model.to(device);

        if optimizer_name.lower() == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=lr)
        elif optimizer_name.lower() == 'rmsprop':
            optimizer = optim.RMSprop(model.parameters(), lr=lr)
        elif optimizer_name.lower() == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)   

        # overfit on a mini dataset (for debugging purposes)
#         ds_train_mini = torch.utils.data.Subset(ds_train, [random.randint(0, len(ds_train) - 1) for _ in range(3)])
#         ds_eval_mini = torch.utils.data.Subset(ds_eval, [0, 1])

        config = TrainerConfig(batch_size=batch_size, epochs=epochs, track_loss=True, checkpoint_path=cp_path)
#         trainer = Trainer(config, model, optimizer, ds_train_mini, ds_eval_mini,
#                           train_collate_fn=train_collate_fn, eval_collate_fn=eval_collate_fn,
#                           metrics_callback_fn=partial(calculate_bleu_score, lang=lang))
        trainer = Trainer(config, model, optimizer, ds_train, eval_ds=ds_eval, test_ds=ds_test, 
                          train_collate_fn=train_collate_fn, eval_collate_fn=eval_collate_fn, 
                          metrics_callback_fn=partial(calculate_bleu_score, lang=lang))
        
        logger.info("Training the model...")
        
        trainer.train()
                
        model, optimizer, trainer, config = None, None, None, None
        gc.collect()
    f.close()

    
if __name__ == '__main__':
    # TODO: these arguments are currently not used. All parameters are loaded from a .yaml file, but it might be useful
    # to allow for specification of arguments on the command line.
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', help='Where to store both the dataset and the experiment results', required=True)
    args = parser.parse_args()
    main(args)
