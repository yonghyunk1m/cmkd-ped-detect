import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassConfusionMatrix, BinaryConfusionMatrix
from torchmetrics.regression import MeanAbsoluteError
from .losses import WeightedOrdinalEMDLoss
from .backbone import MelSpectrogramExtractor
from .vggish import VGGish

class ASPEDLightningModel(pl.LightningModule):
    def __init__(self, token_dim=128, nhead=4, dropout=0.2, num_classes=2,
                 num_layers=1, task_type='classification', learning_rate=5e-4,
                 weight_decay=0.01, class_weights=None, binary_threshold=0.5,
                 backbone_name='vggish', backbone_finetune=False):
        super(ASPEDLightningModel, self).__init__()

        # Ignore class_weights to prevent saving heavy tensors to hyperparameters
        self.save_hyperparameters(ignore=['class_weights'])
        self.num_classes = num_classes
        self.task_type = task_type
        self.binary_threshold = binary_threshold
        self.backbone_name = backbone_name
        self.backbone_finetune = backbone_finetune

        # Register Global Smoothed Weights as a PyTorch buffer
        if class_weights is not None:
            self.register_buffer("class_weights", torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.register_buffer("class_weights", torch.ones(num_classes, dtype=torch.float32))

        # Audio backbone (frozen)
        if backbone_name == 'vggish':
            self.feature_extractor = MelSpectrogramExtractor(n_mels=64)
            self.backbone = VGGish(pretrained=True, freeze=True, per_second=True)
            backbone_dim = 512
        else:
            from .audio_encoders import build_encoder
            self.encoder = build_encoder(backbone_name, finetune=backbone_finetune)
            backbone_dim = self.encoder.output_dim

        self.feature_projection = nn.Linear(backbone_dim, token_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Task-specific setup
        self.val_preds_list = []
        
        if self.task_type == 'regression':
            self.criterion = nn.MSELoss() 
            self.val_metrics = MeanAbsoluteError() 
            self.classifier = nn.Linear(token_dim, 1)
        elif num_classes == 2:
            # Set global pos_weight for binary classification
            pos_weight_val = self.class_weights[1] / (self.class_weights[0] + 1e-8)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val])) 
            self.val_metrics = BinaryConfusionMatrix()
            self.classifier = nn.Linear(token_dim, 1)
        else:
            self.criterion = WeightedOrdinalEMDLoss(num_classes=num_classes)
            self.val_metrics = MulticlassConfusionMatrix(num_classes=num_classes)
            self.classifier = nn.Linear(token_dim, num_classes)

    def train(self, mode=True):
        """Keep frozen backbone in eval mode regardless of model mode."""
        super().train(mode)
        if self.backbone_name == 'vggish':
            self.backbone.eval()
        else:
            self.encoder.eval()
        return self

    def on_load_checkpoint(self, checkpoint):
        """Restore binary BCE pos_weight from saved class_weights buffer."""
        state = checkpoint.get('state_dict', {})
        cw_key = 'class_weights'
        if cw_key in state and self.task_type == 'classification' and self.num_classes == 2:
            cw = state[cw_key]
            pos_weight_val = cw[1] / (cw[0] + 1e-8)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val]))

    def forward(self, x, return_features=False):
        # 1-2. Audio backbone
        if self.backbone_name == 'vggish':
            mel = self.feature_extractor(x)
            backbone_out = self.backbone(mel).transpose(1, 2)  # [B, 10, 512]
        else:
            backbone_out = self.encoder(x)  # [B, 10, D]

        # 3. Transformer and Classifier
        projected_features = self.feature_projection(backbone_out)
        transformer_out = self.transformer(projected_features)

        logits = self.classifier(transformer_out)
        if return_features:
            return logits, transformer_out  # [B,S,n_cls], [B,S,token_dim]
        return logits

    def training_step(self, batch, batch_idx):
        features, labels = batch
        preds = self(features)
        
        if self.task_type == 'regression':
            preds = preds.squeeze(-1).reshape(-1) 
            targets = labels.float().reshape(-1)  
            loss = self.criterion(preds, targets)
            
        elif self.num_classes == 2:
            preds = preds.squeeze(-1).reshape(-1)
            targets = (labels > 0).float().reshape(-1)
            # Uses BCEWithLogitsLoss with predefined global pos_weight
            loss = self.criterion(preds, targets)
                
        else: 
            preds = preds.reshape(-1, self.num_classes) 
            labels_long = labels.long().reshape(-1)     
            
            # Apply predefined Global Smoothed Weights
            sample_weights = self.class_weights[labels_long]
            loss = self.criterion(preds, labels_long, sample_weights=sample_weights)

        self.log('train/loss_step', loss, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        features, labels = batch
        preds = self(features)
        
        if self.task_type == 'regression':
            preds_flat = preds.squeeze(-1).reshape(-1)
            targets_flat = labels.float().reshape(-1)
            loss = self.criterion(preds_flat, targets_flat)
            self.val_metrics.update(preds_flat, targets_flat)
            
        elif self.num_classes == 2:
            preds_flat = preds.squeeze(-1).reshape(-1)
            labels_binary = (labels > 0).float().reshape(-1)
            # Use defined criterion to keep loss scale consistent with training
            loss = self.criterion(preds_flat, labels_binary)
            
            prob = torch.sigmoid(preds_flat)
            pred_labels = (prob > self.binary_threshold).long()
            self.val_preds_list.append(pred_labels.detach())
            self.val_metrics.update(pred_labels, labels_binary.long())
            
        else: 
            preds_flat = preds.reshape(-1, self.num_classes)
            labels_long = labels.long().reshape(-1)
            
            # Apply global weights for validation loss consistency
            sample_weights = self.class_weights[labels_long]
            loss = self.criterion(preds_flat, labels_long, sample_weights=sample_weights)
            
            self.val_preds_list.append(preds_flat.argmax(dim=-1).detach())
            self.val_metrics.update(preds_flat, labels_long)
        
        prefix = 'val_quick' if dataloader_idx == 0 else 'test'
        self.log(f'{prefix}/loss', loss, on_epoch=True, add_dataloader_idx=False)
        return loss

    def _shared_epoch_end(self, prefix):
        if self.task_type == 'regression':
            mae = self.val_metrics.compute()
            self.log(f'{prefix}/mae', mae, prog_bar=True)
            self.val_metrics.reset()
            return

        cm = self.val_metrics.compute().float()
        total_samples = cm.sum()
        
        class_acc = torch.diag(cm) / (cm.sum(dim=1) + 1e-8)
        macro_acc = class_acc.mean()
        self.log(f'{prefix}/macro_accuracy', macro_acc, prog_bar=True)
        for i in range(self.num_classes):
            self.log(f'{prefix}/class_acc_{i}', class_acc[i])
        
        if self.num_classes == 2:
            tn, fp, fn, tp = cm.flatten()
            precision = tp / (tp + fp + 1e-8)
            recall    = tp / (tp + fn + 1e-8)
            f1        = 2 * precision * recall / (precision + recall + 1e-8)
            self.log_dict({
                f'{prefix}/tn': tn, f'{prefix}/fp': fp,
                f'{prefix}/fn': fn, f'{prefix}/tp': tp,
                f'{prefix}/precision': precision,
                f'{prefix}/recall':    recall,
                f'{prefix}/f1':        f1,
            })

        if self.val_preds_list:
            all_preds = torch.cat(self.val_preds_list)
            c_names = ["No_Ped", "Ped_Present"] if self.num_classes == 2 else [f"Class_{i}" for i in range(self.num_classes)]
            for i, name in enumerate(c_names):
                ratio = (all_preds == i).sum().item() / (total_samples + 1e-8) * 100
                self.log(f'{prefix}/pred_ratio_{name}', ratio)

        self.val_preds_list.clear()
        self.val_metrics.reset()

    def on_validation_epoch_end(self):
        self._shared_epoch_end(prefix='val_quick')

    def on_test_epoch_end(self):
        self._shared_epoch_end(prefix='test')

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
    
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx, dataloader_idx=1)