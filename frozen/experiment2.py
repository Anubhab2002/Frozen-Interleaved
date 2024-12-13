import torch
import torch.nn as nn
from torch import optim
import pytorch_lightning as pl
from .model import OPTCaptioningModel
from transformers import GPT2Tokenizer

IMAGE_TOKEN = "<image>"
SPECIAL_TOKEN_DICT = {'additional_special_tokens': [IMAGE_TOKEN]}


class Experiment2(pl.LightningModule):
    def __init__(self, config=dict()):
        super().__init__()
        self.config = config
        self.model = OPTCaptioningModel(config.get('model', dict()))
        self.loss_fn = nn.CrossEntropyLoss()
        
        # Load tokenizer to help with decoding
        text_encoder_name = config.get('model', {}).get('text_encoder', 'facebook/opt-350m')
        self.tokenizer = GPT2Tokenizer.from_pretrained(text_encoder_name)
        if not IMAGE_TOKEN in self.tokenizer.all_special_tokens:
            self.tokenizer.add_special_tokens(SPECIAL_TOKEN_DICT)
        
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
        print(batch_index)
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
        
        # Debug: Print shapes and verify slicing
        # print("Output logits shape:", output.logits.shape)
        # print("Labels shape:", labels.shape)
        
        # Adjust slicing based on actual tensor shapes
        try:
            shift_logits = output.logits[..., N:-1, :].contiguous()
            shift_labels = labels[..., N+1:].contiguous()
        except Exception as e:
            print(f"Error in slicing: {e}")
            print(f"Logits shape: {output.logits.shape}")
            print(f"Labels shape: {labels.shape}")
            raise

        # Compute loss
        loss = self.loss_fn(
            shift_logits.view(-1, V), 
            shift_labels.view(-1)
        )

        # Optional: Generate and log decoded output
        if debug and batch_index % 100 == 0:
            try:
                # Use argmax to generate text
                generated_ids = torch.argmax(output.logits, dim=-1)
                
                # Debugging: print raw token IDs
                print("Generated Token IDs:", generated_ids[0])
                print("Input Token IDs:", kwargs['input_ids'][0])
                
                # Safe decoding
                def safe_decode(tokenizer, token_ids):
                    # Remove padding and special tokens
                    token_ids = [1 if tid > 50265 else tid for tid in token_ids]
                    
                    try:
                        return tokenizer.decode(token_ids, skip_special_tokens=True)
                    except Exception as e:
                        print(f"Decoding error: {e}")
                        return str(token_ids)  # Fallback to showing IDs

                # Decode texts
                generated_text = safe_decode(self.tokenizer, generated_ids[0].cpu().numpy())
                original_text = safe_decode(self.tokenizer, kwargs['input_ids'][0].cpu().numpy())
                label_text = safe_decode(self.tokenizer, labels[0].cpu().numpy())
                
                print("Generated Text:", generated_text)
                print("Original Input Text:", original_text)
                print("Label Text:", label_text)
            except Exception as e:
                print(f"Error in text generation: {e}")
                # Print more detailed error information
                import traceback
                traceback.print_exc()

        # Log the training loss
        self.log('train_loss', loss, on_step=True, prog_bar=True)
        
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
                'lr': 1e-5,
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