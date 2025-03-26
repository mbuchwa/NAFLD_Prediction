from src.utils.helper_functions import *
import torch.nn.functional as F
from torch import optim, nn
from torchmetrics.classification import BinaryAccuracy, Accuracy
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.distributions import Normal
from tab_transformer_pytorch import TabTransformer
import pytorch_lightning as pl


class VI_BNN(nn.Module):
    """Bayesian Neural Network Model"""
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.5, num_classes=2, lr=1e-3, prior_var=1.):
        super().__init__()
        self.hidden_units = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                BLinear(input_dim if i == 0 else hidden_dim,
                        hidden_dim if i < num_layers - 1 else num_classes,
                        prior_var=prior_var),
                nn.Dropout(dropout)  # Dropout layer added
            )
            for i in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
        return torch.softmax(x, dim=1)

    def log_prior(self):
        return sum(layer[0].log_prior for layer in self.hidden_layers)

    def log_post(self):
        return sum(layer[0].log_post for layer in self.hidden_layers)

    def sample_elbo(self, input, target, samples):
        num_classes = self.num_classes
        outputs = torch.zeros(samples, target.shape[0], num_classes).to(input.device)
        log_priors = torch.zeros(samples)
        log_posts = torch.zeros(samples)
        log_likes = torch.zeros(samples)
        for i in range(samples):
            outputs[i] = self(input)
            log_priors[i] = self.log_prior()
            log_posts[i] = self.log_post()
            # since the target is non-binary, so use the cross_entropy
            log_likes[i] = -F.cross_entropy(outputs[i].clone(), target, reduction='sum')

        log_prior = log_priors.mean()
        log_post = log_posts.mean()
        log_like = log_likes.mean()
        loss = log_post - log_prior - log_like
        return loss


class BLinear(nn.Module):
    """Bayesian Base Neural Network"""
    def __init__(self, input_features, output_features, prior_var=1.):
        super().__init__()
        self.w_mean = nn.Parameter(torch.zeros(output_features, input_features))
        self.w_std = nn.Parameter(torch.zeros(output_features, input_features))
        self.b_mean = nn.Parameter(torch.zeros(output_features))
        self.b_std = nn.Parameter(torch.zeros(output_features))
        self.w = None
        self.b = None
        self.prior = Normal(0, prior_var)

    def forward(self, input):
        w_epsilon = Normal(0, 1).sample(self.w_mean.shape).to(get_device(i=0))
        self.w = self.w_mean + torch.log(1 + torch.exp(self.w_std)) * w_epsilon
        b_epsilon = Normal(0, 1).sample(self.b_mean.shape).to(get_device(i=0))
        self.b = self.b_mean + torch.log(1 + torch.exp(self.b_std)) * b_epsilon
        w_log_prior = self.prior.log_prob(self.w)
        b_log_prior = self.prior.log_prob(self.b)
        self.log_prior = torch.sum(w_log_prior) + torch.sum(b_log_prior)
        self.w_post = Normal(self.w_mean.data, torch.log(1 + torch.exp(self.w_std)))
        self.b_post = Normal(self.b_mean.data, torch.log(1 + torch.exp(self.b_std)))
        self.log_post = self.w_post.log_prob(self.w).sum() + self.b_post.log_prob(self.b).sum()
        return F.linear(input, self.w, self.b)


class PLTabTransformer(pl.LightningModule):
    def __init__(self, lr=1e-3, dropout=0.2, num_layers=5, hidden_dim=64, out_dim=2, df_cols=None,
                 classification_type='fibrosis') -> None:
        super().__init__()
        self.lr = lr
        self.dropout = dropout
        self.classification_type = classification_type
        self.num_classes = out_dim

        self.model = TabTransformer(
            categories=[],
            num_continuous=len(df_cols),  # number of continuous values
            dim=hidden_dim,  # dimension, paper set at 32
            dim_out=out_dim,
            depth=num_layers,  # depth, paper recommended 6
            heads=8,  # heads, paper recommends 8
            attn_dropout=0.1,  # post-attention dropout
            ff_dropout=dropout,  # feed forward dropout
            mlp_hidden_mults=(4, 2),  # relative multiples of each hidden dimension of the last mlp to logits
            mlp_act=nn.ReLU(),
        )

    def forward(self, x) -> torch.Tensor:
        x_cont = x
        x_cont = x_cont.to(torch.float32)
        x_cat = torch.tensor([], device=get_device(i=0))  # assume no categorical variables
        x_cat = x_cat.to(torch.int64)
        x = self.model(x_cat, x_cont)
        x = torch.log_softmax(x, dim=1)
        return x

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx) -> float:
        x, y = batch
        x = self(x)
        train_loss = F.nll_loss(x, y)
        preds = torch.argmax(x, dim=1)
        if self.classification_type == 'three_stage':
            train_acc = Accuracy(task='multiclass', num_classes=self.num_classes).to(get_device(i=0))
        else:
            train_acc = BinaryAccuracy().to(get_device(i=0))
        train_acc = train_acc(preds, y)
        self.log('train_loss', train_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', train_acc, on_step=True, on_epoch=True, prog_bar=True)
        return train_loss

    def validation_step(self, batch, batch_idx) -> None:
        x, y = batch
        x = self(x)
        val_loss = F.nll_loss(x, y)
        x = torch.argmax(x, dim=1)
        if self.classification_type == 'three_stage':
            val_acc = Accuracy(task='multiclass', num_classes=self.num_classes).to(get_device(i=0))
        else:
            val_acc = BinaryAccuracy().to(get_device(i=0))
        val_acc = val_acc(x, y)
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', val_acc, on_step=False, on_epoch=True, prog_bar=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = self(x)
        test_loss = F.nll_loss(x, y)
        preds = torch.argmax(x, dim=1)
        if self.classification_type == 'three_stage':
            accuracy = Accuracy(task='multiclass', num_classes=self.num_classes).to(get_device(i=0))
        else:
            accuracy = BinaryAccuracy().to(get_device(i=0))
        accuracy = accuracy(preds, y)
        self.log('test_loss', test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_acc', accuracy, on_step=False, on_epoch=True, prog_bar=True)


class NeuralNetwork(pl.LightningModule):
    def __init__(self, input_dim=1, lr=1e-3, dropout=0.2, num_layers=5, hidden_dim=64, num_classes=2,
                 classification_type='fibrosis') -> None:
        super().__init__()
        self.lr = lr
        self.dropout = dropout
        self.num_classes = num_classes
        self.classification_type = classification_type

        hidden_dims = [hidden_dim] * (num_layers - 1)

        layer_dims = [input_dim] + hidden_dims + [self.num_classes]

        self.layers = nn.ModuleList([nn.Linear(layer_dims[i], layer_dims[i + 1]) for i in range(num_layers)])

    def forward(self, x) -> torch.Tensor:
        x = x.to(self.layers[0].weight.dtype)
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
            x = nn.Dropout(self.dropout)(x)
        x = self.layers[-1](x)
        x = torch.log_softmax(x, dim=1)
        return x

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx) -> float:
        x, y = batch
        x = self(x)
        train_loss = F.cross_entropy(x, y) if self.classification_type == 'three_stage' else F.nll_loss(x, y)
        preds = torch.argmax(x, dim=1)
        if self.classification_type == 'three_stage':
            train_acc = Accuracy(task='multiclass', num_classes=self.num_classes).to(get_device(i=0))
        else:
            train_acc = BinaryAccuracy().to(get_device(i=0))
        train_acc = train_acc(preds, y)
        self.log('train_loss', train_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', train_acc, on_step=True, on_epoch=True, prog_bar=True)
        return train_loss

    def validation_step(self, batch, batch_idx) -> None:
        x, y = batch
        x = self(x)
        val_loss = F.nll_loss(x, y)
        x = torch.argmax(x, dim=1)
        if self.classification_type == 'three_stage':
            val_acc = Accuracy(task='multiclass', num_classes=self.num_classes).to(get_device(i=0))
        else:
            val_acc = BinaryAccuracy().to(get_device(i=0))
        val_acc = val_acc(x, y)
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = self.forward(x)
        test_loss = F.nll_loss(x, y)
        preds = torch.argmax(x, dim=1)
        if self.classification_type == 'three_stage':
            val_acc = Accuracy(task='multiclass', num_classes=self.num_classes).to(get_device(i=0))
        else:
            val_acc = BinaryAccuracy().to(get_device(i=0))
        accuracy = val_acc(preds, y)
        self.log('test_loss', test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_acc', accuracy, on_step=False, on_epoch=True, prog_bar=True)

class DataModule(pl.LightningDataModule):
    def __init__(self, x_train, y_train, x_test, y_test, batch_size=32):
        super().__init__()
        self.x_train = x_train
        self.y_train = y_train

        self.x_test = x_test
        self.y_test = y_test

        self.batch_size = batch_size

    def setup(self, stage=None):
        self.dataset = DataSet(self.x_train, self.y_train)
        self.test_dataset = DataSet(self.x_test, self.y_test)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)


class DataSet(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]