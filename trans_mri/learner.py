import os
import logging
from pathlib import Path
import numpy as np
import pickle
import json

from copy import deepcopy
from dataclasses import dataclass
from functools import partial
import matplotlib.pyplot as plt
from time import time
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import ModuleList

from ignite.handlers import ModelCheckpoint, EarlyStopping
from ignite.engine import Events, Engine
from ignite.metrics import Accuracy, Loss, MetricsLambda, Precision, Recall
from ignite.handlers import EarlyStopping

from .data import DataBunch
from .utils import *
from .models.model_utils import *

if isnotebook():
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm

logger = logging.getLogger(__name__)

class MixLoss: 
    def __init__(self, weights=None, alpha=0.7, **kwargs):
        self.alpha = alpha
        self.loss_mse = nn.MSELoss()
        self.loss_ce = nn.CrossEntropyLoss(weight=weights)
        
    def __call__(self, y_pred, y, x_pred, x):

        loss_mse = self.loss_mse(x_pred, x)
        loss_ce = self.loss_ce(y_pred, y)
        loss = self.alpha * loss_mse + (1-self.alpha) * loss_ce
        return loss

def cond_init(m: nn.Module, init_func):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)

    "Initialize the non-batchnorm layers of `m` with `init_func`"
    if (not isinstance(m, bn_types)) and requires_grad(m):
        if hasattr(m, 'weight'):
            init_func(m.weight)
        if hasattr(m, 'bias') and hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.)


def apply_leaf(m: nn.Module, f):
    "Apply `f` to children of `m`."
    c = children(m)
    if isinstance(m, nn.Module):
        f(m)
    for l in c:
        apply_leaf(l, f)


def apply_init(m, init_func=nn.init.kaiming_normal_):
    "Initialize all non-batchnorm layers of `m` with `init_func`."
    apply_leaf(m, partial(cond_init, init_func=init_func))


def first_layer(m: nn.Module)->nn.Module:
    "Retrieve first layer in a module `m`."
    return flatten_model(m)[0]

def last_layer(m: nn.Module)->nn.Module:
    "Retrieve last layer in a module `m`."
    return flatten_model(m)[-1]

def num_features_model(m: nn.Module)->int:
    "Return the number of output features for a `model`."
    for l in reversed(flatten_model(m)):
        if hasattr(l, 'num_features'):
            return l.num_features

def ndarray_from_tensorbatch(batch: torch.Tensor):
    return torch.squeeze(batch).detach().cpu().numpy()

def add_noise(img, noise_level:float=0.1):
    noise = torch.randn(img.size()) * noise_level
    noisy_img = img + noise
    return noisy_img

def get_non_bias_parameters(model):
    return [p for n,p in model.named_parameters() if "bias" not in n]
  
def _create_trainer(model, model_type, device, optimizer, loss_fn, l1_reg=0.0, **kwargs):

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        x, y = batch
        x,y  = x.to(device),y.to(device)
        if model_type == 'classifier':
            y_pred = model(x)
            loss = loss_fn(y_pred, y)

        elif model_type =='autoencoder':
            x_pred = model(x)
            loss = loss_fn(x_pred, x)
            l1_loss = None
            for w in get_non_bias_parameters(model):
                if l1_loss is None: l1_loss = w.norm(1)
                else: l1_loss = l1_loss + w.norm(1)
            loss = loss + l1_reg * l1_loss                
            engine.x_pred = x_pred

        elif model_type =='mixed':
            x_pred, y_pred = model(x)
            loss = loss_fn(y_pred, y, x_pred, x)
            if isinstance(loss, tuple) and len(loss) == 3:
                loss, loss_mse, loss_ce = loss
            engine.x_pred = x_pred

        loss.backward()
        optimizer.step()
        return loss.item()

    return Engine(_update)

def _create_evaluator(model, model_type, device, metrics={}):
    
    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            x, y = batch
            x,y  = x.to(device),y.to(device)
            if model_type == 'classifier':
                y_pred = model(x)
                return y_pred, y
            elif model_type == 'autoencoder':
                x_pred = model(x)
                return x_pred, x
            elif model_type=='mixed':
                x_pred, y_pred = model(x)
                return {'x': x, 'y': y, 'x_pred': x_pred, 'y_pred': y_pred}

    engine = Engine(_inference)
    for name, metric in metrics.items():
        metric.attach(engine, name)
    return engine

def _create_predictor(model, device):
    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            x, _ = batch
            x = x.to(device)
            pred = model(x)
            engine.pred = pred
            return pred
    return Engine(_inference)

@dataclass
class Learner():
    """Learner object connecting DataBunch with a model, enabling training.
    This class also logs metadata, prints results, saves trained models/encoders, 
    implements methods to visualize training and latent manifolds.

    Important methods:
    - fit: fit model on the training set in data
    - from_pretrained: classmethod to initialize a Learner from a pretrained torch model (in pickle format).
    - predict: returrns predictions and confusion matrix of the test set
    - plot_history: plot training curves
    - plot_bottleneck_histogram: plot distribution of activations of the code
    - plot_TSNE: compute and plot the 2D tSNE projection of the test set in latent space
    - freeze_encoder/unfreeze_encoder: freeze/unfreeze the encoder
    - reset_model: delete saved pickle and re-initialize model weights
    - save: saves model to path/model_dir
    

    # Arguments:
        data: Databunch object containing train and test datasets.
        model: PyTorch model architecture to train.
        path: Path where intermediary/metadata files will be written, eg. models, encoders, history, etc.
        loss_fn: PyTorch loss function (callable), by default defined by model_type.
        opt_fn: A torch.optim callable, by default using the Adam optimizer.
        device: Cuda device or 'cpu' used for computation.
        model_dir: Directory relative to path to store models.
        tmp_dir: Directory relative to path to store visualizations, history, etc.
        model_type: Can be {classifier, autoencoder, mixed}. Ny default, classifiers use CrossEntropyLoss, 
        autoencoders MSELoss (+regularization, optional) and mixed models' objective is both reconstruction and classification.
        bs: Batch size used for iterators.
    """
    data: DataBunch
    model: nn.Module
    path: str = None
    loss_fn: callable = None
    opt_fn: callable = optim.Adam
    device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_dir: str = "models"
    tmp_dir: str = "tmp"
    model_type: str = "classifier"
    bs: int = 8

    def __post_init__(self):
        self.path = self.data.path if self.path is None else self.path
        self.path = Path(self.path)
        self.model_dir = self.path/self.model_dir
        (self.model_dir).mkdir(parents=True, exist_ok=True)
        self.tmp_dir = self.path/self.tmp_dir
        (self.tmp_dir).mkdir(parents=True, exist_ok=True)
        self.device = self.data.device if self.device is None else self.device

        self.model = self.model.to(self.device)
        self.model_name = self.model.__class__.__name__

        if self.data.train_dl is None:
            self.data.build_dataloaders(bs=self.bs, normalize=False, use_samples=None)

        self.bs = self.data.train_dl.batch_size
        if not torch.cuda.is_available():
            print(f"WARNING: No cuda device is available, using {self.device}")
        # self.init_trainer()
        print(f"\n{self.model_type} learner initialized at {str(self.path)}")
    
    @classmethod
    def from_pretrained(cls, data: DataBunch, model: nn.Module, path: str,
                        pretrained_model: str, model_type:str, 
                        loss_fn: callable=None, opt_fn: callable=optim.Adam, 
                        device: torch.device = None):
        print(f"Initializing Learner from pretrained model at {pretrained_model}")
        learn = cls(data, model, path, loss_fn=loss_fn, opt_fn=opt_fn,
                    model_type=model_type, device=device)
        
        if os.path.isfile(pretrained_model) and str(pretrained_model).endswith('.pth'):
            learn.load(pretrained_model, map_location=device)
            learn.model.to(device)
            return learn
        elif not os.path.isfile(pretrained_model):
            raise IOError(f"Pretrained model not existing at {pretrained_model}")
        elif not str(pretrained_model).endswith('.pth'):
            raise ValueError(f"Pretrained model has to be a .pth file.")
    
    def init_trainer(self, l1_reg=0.0, **kwargs):

        if self.model_type == "classifier":
            self.history = {'train_loss': [], 'train_f1': [], 
                            'valid_loss': [], 'valid_f1': []}
            weights = torch.FloatTensor(self.data.train_ds.class_weights)
            if torch.cuda.is_available(): weights = weights.to(self.device)
            self.loss_fn = nn.CrossEntropyLoss(weight=weights)

            precision = Precision(average=False)
            recall = Recall(average=False)
            def Fbeta(r, p, beta):
                return torch.mean((1 + beta ** 2) * p * r / (beta ** 2 * p + r + 1e-20)).item()

            self.metrics = {'acc': Accuracy(), 'loss': Loss(self.loss_fn), 
                            'f1': MetricsLambda(Fbeta, recall, precision, 1)}

        elif self.model_type == "autoencoder":
            self.history = {'train_loss': [], 'valid_loss': []}

            self.loss_fn = nn.MSELoss()
            self.metrics = {'loss': Loss(self.loss_fn)}

        elif self.model_type == "mixed":
            self.history = {'train_loss': [], 'valid_loss': []}

            def output_transform(output):
                x, y = output['x'], output['y']
                x_pred, y_pred = output['x_pred'], output['y_pred']
                return y_pred, y, {'x_pred': x_pred, "x": x}

            weights = torch.FloatTensor(self.data.train_ds.class_weights)
            if torch.cuda.is_available(): weights = weights.to(self.device)

            self.loss_fn = MixLoss(weights=weights,**kwargs)
            self.metrics = {'loss': Loss(self.loss_fn, output_transform=output_transform)}

        else:
            raise ValueError(f"Attribute `model_type`={self.model_type} is not valid.")

        self.test_metrics = {}
        self.trainer = _create_trainer(self.model, self.model_type, self.device, self.opt_fn,
                                       self.loss_fn, l1_reg=l1_reg, **kwargs)

        self.evaluator = _create_evaluator(self.model, self.model_type, self.device, metrics=self.metrics)

    def init_handlers(self, kwargs):
        with_valid = kwargs.get("with_valid")
        show_every = kwargs.get("show_every")
        save_fig = kwargs.get("save_fig")
        show_fig = kwargs.get("show_fig")
        early_stop = kwargs.get("early_stop")
        log_interval = kwargs.get("log_interval")
        save_interval = kwargs.get("save_interval")
        model_prefix = kwargs.get("model_prefix")
        debug = kwargs.get("debug")
        num_iters = len(self.data.train_dl)


        def score_early_stop(engine):
            val_loss = engine.state.metrics['loss']
            return -val_loss

        def log_training_loss(engine):
            iter = (engine.state.iteration - 1) % len(self.data.train_dl) + 1
            if log_interval is None or log_interval == 0: return
            if iter % log_interval == 0:
                print(f"epoch {engine.state.epoch} iter: {iter}/{num_iters}"
                      f" loss: {round(engine.state.output, 3)}", end="\r")

        def log_training_results(engine):
            self.evaluator.run(self.data.train_dl)
            metrics = self.evaluator.state.metrics
            epoch_i, num_epochs = engine.state.epoch, engine.state.max_epochs
            epoch_i = "0"+str(epoch_i) if len(str(epoch_i)) < len(str(num_epochs)) else str(epoch_i)
            log = "[{}/{}]  train_loss: {:.3f}".format(epoch_i, num_epochs, metrics['loss'])

            if self.model_type=='classifier':
                log+=" - f1: {:.3f}".format(metrics['f1'])
                log+=" - acc: {:.3f}".format(metrics['acc'])
                self.history['train_f1'].append(metrics['f1'])
                
            self.history['train_loss'].append(metrics['loss'])

            if with_valid: print(log, end=' ')
            else: print(log)

        def log_test_results(engine):
            self.evaluator.run(self.data.test_dl)
            metrics = self.evaluator.state.metrics
            print()
            print(f"Evaluation on test set:")
            log = "loss: {:.4f}".format(metrics['loss'])
            self.test_metrics['loss'] = metrics['loss']
            if self.model_type=='classifier':
                log +="\nacc: {:.3f} \nf1: {:.3f}".format(metrics['acc'], metrics['f1'])
                self.test_metrics['acc'] = metrics['acc']
                self.test_metrics['f1'] = metrics['f1']
            
            print(log)

        def log_valid_results(engine):
            self.evaluator.run(self.data.test_dl)
            metrics = self.evaluator.state.metrics
            log = "- valid_loss: {:.3f}".format(metrics['loss'])
            if self.model_type=='classifier':
                log+=" - f1: {:.3f}".format(metrics['f1'])
                log+=" - acc: {:.3f}".format(metrics['acc'])

                self.history['valid_f1'].append(metrics['f1'])

            self.history['valid_loss'].append(metrics['loss'])

            print(log)

        def plot_training_reconstruction(engine):
            # TODO: VISUALIZE THE AVERAGE IMAGE IN EPOCH
            epoch = engine.state.epoch
            if epoch % show_every == 0:
                original, _ = engine.state.batch
                reconstr = engine.x_pred
                original = ndarray_from_tensorbatch(original)
                reconstr = ndarray_from_tensorbatch(reconstr)

                if original.ndim > 3: original = original[0, ...]
                if reconstr.ndim > 3: reconstr = reconstr[0, ...]

                loss = np.round(
                    F.mse_loss(
                        torch.from_numpy(original),torch.from_numpy(reconstr)).item(), 4)

                title = f"epoch={epoch}, loss_mse={loss}"
                f = plot_original_reconstructed(original, reconstr, title=title)
               

                fig_path = str(self.tmp_dir/f"reconstruction_{epoch}.png")
                if show_fig: plt.show(f)
                if save_fig: f.savefig(fig_path, dpi=300); plt.close(f)


        # ADD EVENT HANDLERS TO TRAINER
        self.trainer.add_event_handler(Events.ITERATION_COMPLETED, log_training_loss)
        self.trainer.add_event_handler(Events.EPOCH_COMPLETED, log_training_results)
        if len(self.data.test_dl) > 0:
            self.trainer.add_event_handler(Events.COMPLETED, log_test_results)
        
        if save_interval is not None:
            model_prefix = get_timestamp()+"_"+model_prefix
            checkpointer = ModelCheckpoint(self.model_dir,
                                    model_prefix,
                                    save_interval=save_interval,
                                    n_saved=1,
                                    create_dir=True)
            self.trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {self.model_name: self.model})
            print(f"Model checkpointing at every {save_interval} epochs to {self.model_dir}")

        if early_stop is not None:
            early_stopper = EarlyStopping(patience=early_stop, score_function=score_early_stop, trainer=self.trainer)
            self.evaluator.add_event_handler(Events.COMPLETED, early_stopper)
            print(f"Early stopping is active with patience {early_stop}")

        if with_valid: 
            # self.history['valid_loss'] = []                    
            # if self.model_type =='classifier': self.history['valid_acc'] = []

            self.trainer.add_event_handler(Events.EPOCH_COMPLETED, log_valid_results)

        if show_every is not None:
            self.trainer.add_event_handler(Events.EPOCH_COMPLETED, plot_training_reconstruction)
        
    def create_opt(self, lr, wd, num_epochs):
            if callable(self.opt_fn): self.opt_fn_callable = deepcopy(self.opt_fn)
            else: self.opt_fn = self.opt_fn_callable
            
            self.opt_fn = self.opt_fn(filter(lambda p: p.requires_grad, self.model.parameters()),
                                 lr=lr, weight_decay=wd)
        
    def fit(self, num_epochs: int, lr:float=1e-3, wd:float=0.0, l1_reg:float=0.0,
            with_valid:bool=True, show_every:int=None, save_fig:bool=True, show_fig:bool=True,
            log_interval:int=10, save_interval:int=None, early_stop:int=None,
            model_prefix:str="model", **kwargs):
        """Fits model on train data for num_epochs with the passed arguments, optionally using test set as validation.
        After training, model/encoder are saved to path/model_dir, as well as the training history.

        # Arguments:
            num_epochs: Number of epochs to train for.
            lr: Learning rate used to for gradient descent.
            wd: Weight-decay parameter (or L1 penalty).
            l1_reg: L1 penalty parameter.
            with_valid: Whether to use test set to approximate generalization performance after each training epoch.
            show_every: Show reconstructed images in every show_every epochs (for autoencoders).
            save_fig: Whether to save reconstructed images to tmp_dir.
            show_fig: Whether to show reconstructed images.
            log_interval: Logging metrics in every log_interval iterations.
            save_interval: Saving model after every save_interval epochs.
            early_stop: Early stopping patience, interrupting training if val_loss is not decreasing.
            model_prefix: Saved model's prefix
        """
        self.model.to(self.device)
        self.create_opt(lr, wd, num_epochs)
        self.init_trainer(l1_reg=l1_reg, **kwargs)
        self.init_handlers({'log_interval': log_interval,
                            'save_interval': save_interval,
                            'early_stop': early_stop,
                            'show_every': show_every,
                            'save_fig': save_fig,
                            'show_fig': show_fig,
                            'model_prefix': model_prefix,
                            'with_valid': with_valid})

        if num_epochs > 0:
            t0 = time()
            print(f"Fitting {self.model_type} architecture `{self.model_name}`"
                  f" for {num_epochs} epochs\n")
            self.trainer.run(self.data.train_dl, max_epochs=num_epochs)
            with open(self.tmp_dir/'history.json', 'w') as fp:
                json.dump(self.history, fp, indent=4)
            with open(self.tmp_dir/'test_metrics.json', 'w') as fp:
                json.dump(self.test_metrics, fp, indent=4)

            time_elapsed = str(timedelta(seconds=time()-t0))
            print("Total fit time:", time_elapsed)

    def predict(self):
        preds = []
        def get_predictions(engine):
            preds.extend(torch.squeeze(engine.pred).detach().cpu().numpy())

        predictor = _create_predictor(self.model, self.device)
        predictor.add_event_handler(Events.ITERATION_COMPLETED, get_predictions)
        predictor.run(self.data.test_dl)
        preds = np.array(preds).argmax(axis=1)
        cm =  confusion_matrix(self.data.test_ds.labels.numpy(), preds)
        plot_confusion_matrix(cm, self.data.classes, normalize=False)
        return preds, cm

    def encode_data(self, to_encode, device='gpu'):
        assert to_encode in ["train", "test"], print("argument to_encode has to be train or test.")
        enc_str = "encoded_"+to_encode
        if hasattr(self, enc_str):
            print(f"WARNING: {to_encode} data already encoded, using cache")
            X, Y = getattr(self, enc_str)
        
        else:
            dl = self.data.train_dl if to_encode=='train' else self.data.test_dl
            X, Y = [], []
            for x,y in tqdm(dl, desc=f'encoding {to_encode} data'):
                
                device = torch.device("cuda", device) if isinstance(device, int) else device
                device = self.device if device=="gpu" else device

                if isinstance(device, torch.device):
                    x = x.to(device)
                    self.model = self.model.to(device)
                
                elif device=='cpu':
                    self.model = self.model.cpu()
                
                else:
                    raise ValueError("Argument device has to be one of [cpu,gpu].")
                
                x_enc = self.model.encode(x)
                
                x_enc = x_enc.view(x.size(0), -1).detach().cpu().numpy()
                X.extend(x_enc)
                Y.extend(y.detach().cpu().numpy())
            X, Y = np.array(X), np.array(Y)
            setattr(self, enc_str, [X, Y])
        return X, Y

    def plot_history(self, show=True, **kwargs):
        history = self.history
        n = len(history['train_loss'])
        x = np.arange(1, n+1)
        f, axs = plt.subplots(nrows=1, ncols=1, figsize = (10,5))
        axs.plot(x, history['train_loss'], label='train loss')
        if history.get('valid_loss') is not None :
            if len(history['valid_loss']) > 0: 
                axs.plot(x, history['valid_loss'], label='valid_loss')
        axs.set_xticks(x)
        axs.set_xlabel('epochs')
        axs.set_ylabel('loss')
        axs.legend()
        if show: plt.show()
        f.savefig(self.tmp_dir/'history.png')

    def plot_bottleneck_histogram(self, data:str='test', show:bool=False, device:str='gpu', 
                                  title:str=None, file_name:str=None, **kwargs):
        X, _ = self.encode_data(data, device=device)
        title = f'bottleneck histrogram of {data} data' if title is None else title

        f, ax = plt.subplots(1,1)
        ax.hist(X.flatten(), bins = 100)
        ax.set_xlabel('activation')
        ax.set_title(title)
        if show: plt.show()
        file_name = f'{data}_bottleneck_hist.png' if file_name is None else file_name
        f.savefig(self.tmp_dir/file_name, dpi=300)

    def plot_TSNE(self, data:str='test', show:bool=True, device:str='gpu', 
                  title:str=None, file_name:str=None, **kwargs):
        X_enc, y = self.encode_data(data, device=device)
        tsne = TSNE(n_components=2, init='pca',early_exaggeration=12, random_state=0)
        X_tsne = tsne.fit_transform(X_enc)
        title = f't-SNE projection of {data} data' if title is None else title
        f, ax = plot_projection(X_tsne, y, title=title, labels=self.data.id2label)
        if show: plt.show()
        file_name = f'{data}_tsne.png' if file_name is None else file_name
        f.savefig(self.tmp_dir/file_name, dpi=300)

    def plot_PCA(self, data:str='test', show:bool=True, device:str='gpu', 
                 title:str=None, file_name:str=None, **kwargs):
        X_enc, y = self.encode_data(data, device=device)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_enc)
        title = f'PCA projection of {data} data' if title is None else title
        f, ax = plot_projection(X_pca, y, title=title, labels=self.data.id2label)
        if show: plt.show()
        file_name = f'{data}_pca.png' if file_name is None else file_name
        f.savefig(self.tmp_dir/file_name, dpi=300)

    def unfreeze(self):
        pass

    def freeze(self):
        pass

    def freeze_to(self, n: int):
        pass

    def __del__(self): del(self.model, self.data)

    def reset_model(self):
        init_weights(self.model)
        if os.path.exists(self.model_dir/'model.pth'):
            os.remove(self.model_dir/'model.pth')
        print("Learner has ben reset (model re-initialized, model.pth deleted).")

    def save_encoder(self, name:str='encoder'):
        file_name = name+'.pth'
        if hasattr(self.model, "encoder"):
            torch.save(self.model.encoder,
                       self.model_dir/file_name)
            print(f"Encoder has been saved to {self.model_dir/file_name}")
        return str(self.model_dir/file_name)

    def save(self, name:str='model'):
        "Save model with `name` to `self.model_dir`."

        file_name = name+'.pth' if not name.endswith('.pth') else name
        torch.save(self.model.state_dict(), self.model_dir/file_name)
        print(f"Model has been saved to {self.model_dir/file_name}")

    def load(self, name:str, map_location=None):
        name = str(name)
        "Load model `name` from `self.model_dir`."
        file_name = name+'.pth' if not name.endswith('.pth') else name
        self.model.load_state_dict(torch.load(self.model_dir/file_name, map_location=map_location))

    def unfreeze_encoder(self):
        unfreeze_encoder(self.model)

    def freeze_encoder(self):
        freeze_encoder(self.model)

def unfreeze_encoder(model):
    for name, child in model.named_children():
        if name == "encoder":
            for param in child.parameters():
                param.requires_grad = True
    print(f"Encoder unfreezed (no. trainable params: {get_num_params(model)},"
          f" encoder: {get_num_params(model.encoder)}; classifier: {get_num_params(model.classifier)})")

def freeze_encoder(model):
    for name, child in model.named_children():
        if name == "encoder":
            for param in child.parameters():
                param.requires_grad = False
    print(f"encoder freezed (no. trainable params: {get_num_params(model)},"
          f" encoder: {get_num_params(model.encoder)}; classifier: {get_num_params(model.classifier)})")

def create_classifier(data:DataBunch, model:nn.Module, path=None, **kwargs):
    """Factory function to create classifier Learner"""
    path = data.path if path is None else path
    return Learner(data, model, path, model_type="classifier", **kwargs)

def create_autoencoder(data:DataBunch, model:nn.Module,
                       path=None, alpha=0.6, model_type='autoencoder', **kwargs):
    """Factory function to create autoencoder Learner"""
    assert model_type in ['autoencoder', 'mixed'], "Argument `model_type` has to be either mixed or autoencoder!"
    if model_type == 'mixed' and not hasattr(model, "classifier"):
        raise AttributeError("Attribute classifier not found."
                             "When using `mixed` model_type, make sure your model has a `classifier` attribute.")
    path = data.path if path is None else path
    return Learner(data, model, path, model_type=model_type,
                   **kwargs)

def create_classifier_from_encoder(data:DataBunch, encoder_path:str=None, path=None,  
                                   dropout1=0.5, device: torch.device = torch.device('cuda', 0), **kwargs):
    """Factory function to create classifier from encoder to allow transfer learning."""
    from .models.models import EncoderClassifier
    path = data.path if path is None else path
    if encoder_path is None:
        logger.info("WARNING: `encoder_path` is None, not using pretrained feature extractor")
        encoder = None
    else:
        encoder = torch.load(encoder_path, map_location='cpu')
    model = EncoderClassifier(data.train_ds.shape, encoder, len(data.classes),dropout1=dropout1)
    learn =  Learner(data, model, path, model_type="classifier", device=device, **kwargs)
    learn.freeze_encoder()
    return learn

