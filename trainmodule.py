import torch.optim as optim
import pytorch_lightning as pl
import torch
import torch.nn as nn

# from adabelief_pytorch import AdaBelief
from torchmetrics import CharErrorRate

class LASTrainModule(pl.LightningModule):
    def __init__(self, model, max_label_len, label_smoothing, tf_rate_lower, tf_rate_upper, tf_decay_step, fn_int_to_text):
        super().__init__()
        self.loss_fn = torch.nn.CrossEntropyLoss(weight=None, label_smoothing=label_smoothing, ignore_index=0)
        self.model = model
        self.valid_metrics = CharErrorRate()
        self.max_label_len = max_label_len
        self.tf_rate_lower = tf_rate_lower
        self.tf_rate_upper = tf_rate_upper
        self.tf_decay_step = tf_decay_step
        self.model.update_tf_rate(tf_rate_upper)
        self.ind2txt = fn_int_to_text
    
    def cal_df_rate(self, epoch):
        return self.tf_rate_upper - (self.tf_rate_upper-self.tf_rate_lower)*min((float(epoch)/self.tf_decay_step),1)

    def forward(self, *args):
        return self.model(*args)

    def training_step(self, batch, batch_idx):
        data, label = batch
        max_label_len = min([label.size()[1], self.max_label_len])
        pred_y = self(data, label)
        pred_y = torch.stack(pred_y).transpose(0,1)[:,:max_label_len,:]
        #pred_y = (torch.cat([torch.unsqueeze(each_y,1) for each_y in pred_y],1)[:,:max_label_len,:]).contiguous()
        #pred_y = pred_y.permute(0,2,1)#pred_y.contiguous().view(-1,output_class_dim)
        # true_y = nn.functional.one_hot(label.long(), num_classes=num_class).type(torch.float)[:, :max_label_len]
        #true_y = torch.argmax(label,dim=2)[:, :max_label_len]#.view(-1)
        true_y = label[:, :max_label_len].type(torch.float)
        true_y_index = torch.argmax(true_y, dim=2)
        loss = self.loss_fn(pred_y.permute(0,2,1), true_y_index)
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.model.update_tf_rate(self.cal_df_rate(self.global_step))
        return loss

    def validation_step(self, batch, batch_idx):
        data, label = batch
        max_label_len = min([label.size()[1], self.max_label_len])
        pred_y = self(data, label)
        pred_y = torch.stack(pred_y).transpose(0,1)[:,:max_label_len,:]
        #pred_y = (torch.cat([torch.unsqueeze(each_y,1) for each_y in pred_y],1)[:,:max_label_len,:]).contiguous()
        #pred_y = pred_y.permute(0,2,1)#pred_y.contiguous().view(-1,output_class_dim)
        # true_y = nn.functional.one_hot(label.long(), num_classes=num_class).type(torch.float)[:, :max_label_len]
        true_y = label[:, :max_label_len].type(torch.float)
        #true_y = torch.max(label,dim=2)[1][:, :max_label_len].contiguous()#.view(-1)
        true_y_index = torch.argmax(true_y, dim=2)
        pred_y_index = torch.argmax(pred_y, dim=2).cpu().tolist()
        true_txt = [self.ind2txt(y) for y in true_y_index.cpu().tolist()]
        pred_txt = [self.ind2txt(y) for y in pred_y_index]

        loss = self.loss_fn(pred_y.permute(0,2,1), true_y_index)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.valid_metrics.update(pred_txt, true_txt)

    def validation_epoch_end(self, outputs):
        cer = self.valid_metrics.compute()
        self.log("val_cer", cer.item())
        self.valid_metrics.reset()
        #self.model.update_tf_rate(self.cal_df_rate(self.trainer.current_epoch+1))


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3, amsgrad=True)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=1, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }
