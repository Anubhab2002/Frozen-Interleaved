import torch
import torch.nn as nn
from torch import optim
import pytorch_lightning as pl
from .model import OPTCaptioningModel


class Experiment(pl.LightningModule):
    def __init__(self, config=dict()):
        super().__init__()
        self.config = config
        self.model = OPTCaptioningModel(config.get('model', dict()))
        self.loss_fn = nn.CrossEntropyLoss()
        self.save_hyperparameters(config)

    @property
    def model_config(self):
        return self.model.config

    @property
    def num_image_tokens(self):
        return self.model_config.get('num_image_tokens')

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)


    def training_step(self, batch, batch_index, debug=False):
        V = self.model.text_encoder.config.vocab_size  # Vocabulary size
        N = self.num_image_tokens  # Number of image tokens

        # Move inputs to the correct device
        kwargs = {
            'pixel_values': batch['pixel_values'].to('cuda:0'),
            'input_ids': batch['input_ids'].to('cuda:0'),
            'attention_mask': batch['attention_mask'].to('cuda:0'),
            'image_token_mask': batch['image_token_mask'].to('cuda:0'),
        }

        # Forward pass
        output = self.forward(**kwargs)

        # Compute the loss using suffix targets
        labels = batch['labels']
        # print('logits: ', output.logits, output.logits.shape)
        # print('labels: ', labels.shape)
        shift_logits = output.logits[..., N:-1, :].contiguous()
        shift_labels = labels[..., N+1:].contiguous()

        # print(shift_logits.shape, shift_labels.shape)
        
        loss = self.loss_fn(shift_logits.view(-1, V), shift_labels.view(-1))

        # Log the training loss
        self.log('train_loss', loss, on_step=True)
        
        return {'loss': loss}


    @torch.no_grad()
    def validation_step(self, batch, batch_index):
        return self.training_step(batch, batch_index)

    def validation_epoch_end(self, outputs):
        print("validation done...........")
        val_loss = torch.stack([x['loss'] for x in outputs]).mean()
        val_perplexity = val_loss.exp()
        self.log('val_loss', val_loss)
        self.log('val_perplexity', val_perplexity, prog_bar=True)

    @property
    def optimizer(self):
        default = {
            'algorithm': 'Adam',
            'params': {
                'lr': 0.000003,
                'betas': [0.9, 0.95],
            },
        }

        return self.config.get('optimizer', default)

    def configure_optimizers(self):
        method = eval(f"optim.{self.optimizer['algorithm']}")
        params = self.optimizer['params']
        parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = method(parameters, **params)

        return {
            'optimizer': optimizer,
        }