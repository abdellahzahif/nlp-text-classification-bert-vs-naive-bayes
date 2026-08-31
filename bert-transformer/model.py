from pytorch_lightning import LightningModule
from torch.optim import Adam
import config as CONFIG



class RecipERT(LightningModule):
    '''
    your model, aka "RecipERT". RecipERT ist a multi-label classifier which - given a recipe -
    returns multiple categories for the recipe.
    '''

    def __init__(self, ncategories):
        super(RecipERT, self).__init__()
        # FIXME

    def forward(self, tokens, attn_mask, return_logits=True):
        '''
        your model's forward pass.
        @param tokens: a batch of tokens, i.e. a tensor of shape BATCHSIZE x SEQUENCELENGTH
        @type tokens: torch.tensor(long)
        @attn_mask: the attention mask for feeding to the transformer
                    (required to ignore padding tokens in attention).
        @type attn_mask: torch.tensor(long)
        @param return_logits: a flag indicating if the model should return logits (logits=True)
                       or probabilities (logits=False).
        @type return_logits: boolean
        '''
        # FIXME
        return scores


    def training_step(self, batch, batch_idx):
        '''
            implement the computation of the loss for a training batch here.
            @param batch: a batch of training samples, of the form

            {
                'tokens': tokens,
                'labels': labels,
                'attn_mask': attn_mask
            }

            where
            - tokens is a tensor of tokens (shape BATCHSIZE x SEQUENCE_LENGTH).
            - labels is a tensor of class labels (shape BATCHSIZE x #CLASSES).
            - attn_mask is a tensor of tokens (shape BATCHSIZE x SEQUENCE_LENGTH).
        '''
        tokens,labels,attn_mask = batch['tokens'],batch['labels'],batch['attn_mask']
        # FIXME
        # remark: optionally, you can log the loss to tensorboard or wandb here.
        return loss


    def validation_step(self, batch, batch_idx):
        '''
            implement the computation of the loss for a validation batch here (see parameters
            above, in 'training step').
        '''
        tokens,labels,attn_mask = batch['tokens'],batch['labels'],batch['attn_mask']
        # FIXME
        # remark: optionally, you can log the loss to tensorboard or wandb here.
        return loss


    def test_step(self, batch, batch_idx):
        raise NotImplementedError()


    def configure_optimizers(self):
        return Adam(self.parameters(), lr=CONFIG.LR)

