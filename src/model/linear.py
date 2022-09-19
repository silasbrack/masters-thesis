import torch
from laplace import Laplace, ParametricLaplace
from pytorch_lightning import LightningModule
from torch import nn
from torch.nn import MSELoss
from torch.optim import Adam


class LinearLightningModule(LightningModule):
    def __init__(
        self,
        lr: float,
        num_layers: int,
        latent_size: int,
        num_features: int,
        hessian_structure: str,
        optimize_prior_precision: bool,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.latent_size = latent_size
        self.num_features = num_features
        self.lr = lr
        self.hessian_structure = hessian_structure if isinstance(hessian_structure, list) else [hessian_structure]
        self.optimize_prior_precision = (
            optimize_prior_precision if isinstance(optimize_prior_precision, list) else [optimize_prior_precision]
        )
        self.loss = MSELoss()
        self.save_hyperparameters()

        layers = self._get_layers()
        self.model = nn.Sequential(*layers)

    def _get_layers(self):
        layers = []
        for i in range(self.num_layers):
            size_in = self.num_features if i == 0 else self.latent_size  # If first layer
            size_out = 1 if i == self.num_layers - 1 else self.latent_size  # If last layer

            layers.append(nn.Linear(size_in, size_out))

            if i != self.num_layers - 1:
                layers.append(nn.ReLU())
        return layers

    def forward(self, x):
        """method used for inference input -> output."""
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """needs to return a loss from a single batch."""
        _, loss = self.step(batch)
        self.log("train/loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """used for logging metrics."""
        preds, loss = self.step(batch)
        self.log("val/loss", loss, sync_dist=True)

        return preds

    def test_step(self, batch, batch_idx):
        """used for logging metrics."""
        _, loss = self.step(batch)
        self.log("test/loss", loss, sync_dist=True)

    def on_train_epoch_end(self):
        """At the end of every epoch, log the Hessian."""
        for opt in self.optimize_prior_precision:
            for struct in self.hessian_structure:
                la: ParametricLaplace = Laplace(
                    self.model,
                    likelihood="regression",
                    subset_of_weights="all",
                    hessian_structure=struct,
                )
                la.fit(self.trainer.train_dataloader)
                if opt:  # self.optimize_prior_precision:
                    opt_str = "opt"
                    la.optimize_prior_precision()
                else:
                    opt_str = "unopt"
                self.log("prior_precision", la.prior_precision, sync_dist=True)
                self.log("mean_posterior_precision", la.posterior_precision.mean(), sync_dist=True)

                torch.save(
                    la.posterior_precision,
                    f"posterior_precision_epoch{self.trainer.current_epoch}_{opt_str}_{struct}.pt",
                )
                torch.save(la.H, f"hessian_epoch{self.trainer.current_epoch}_{opt_str}_{struct}.pt")
                # self.log("hessian", la.H, sync_dist=True)

    def configure_optimizers(self):
        """defines model optimizer."""
        return Adam(self.parameters(), lr=self.lr)

    def step(self, batch):
        """convenience function since train/valid/test steps are similar."""
        x, y = batch
        preds = self(x)
        loss = self.loss(preds, y)
        return preds, loss
