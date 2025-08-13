import argparse, os
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger
import shutil
import json
import pytorch_lightning as pl
from torch.optim import AdamW, Adam, SGD
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler
from collections import Counter
from scipy.stats import norm
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

##import user lib
from base.data_eeg import load_eeg_data
from base.data_meg import load_meg_data
from base.utils import update_config , ClipLoss, instantiate_from_config, get_device

device = get_device('auto')

def load_model(config,train_loader,test_loader):
    model = {}
    for k,v in config['models'].items():
        print(f"init {k}")
        model[k] = instantiate_from_config(v)

    pl_model = PLModel(model,config,train_loader,test_loader)
    return pl_model

class PLModel(pl.LightningModule):
    def __init__(self, model,config,train_loader,test_loader, model_type = 'RN50'):
        super().__init__()
        # savinf traning results
        # self.eeg_outputs=[]
        # self.img_outputs=[]

        self.config = config
        for key, value in model.items():
            setattr(self, f"{key}", value)
        self.criterion = ClipLoss()

        self.all_predicted_classes = []
        self.all_true_labels = []
    
        self.z_dim = self.config['z_dim']

        self.sim = np.ones(len(train_loader.dataset))
        self.match_label = np.ones(len(train_loader.dataset), dtype=int)
        self.alpha = 0.05
        self.gamma = 0.3
        
        self.mAP_total = 0
        self.match_similarities = []
        

    def forward(self, batch,sample_posterior=False):
 
        idx = batch['idx'].cpu().detach().numpy() 
        eeg = batch['eeg']
        img = batch['img']

        img_z  = batch['img_features']
        
        eeg_z = self.brain(eeg)
        img_z = img_z/img_z.norm(dim=-1, keepdim=True)

        logit_scale = self.brain.logit_scale
        logit_scale = self.brain.softplus(logit_scale)
        
        eeg_loss, img_loss, logits_per_image = self.criterion(eeg_z, img_z, logit_scale)
        total_loss = (eeg_loss.mean() + img_loss.mean()) / 2

        if self.config['data']['uncertainty_aware']:
            diagonal_elements = torch.diagonal(logits_per_image).cpu().detach().numpy()
            gamma = self.gamma

            batch_sim = gamma * diagonal_elements + (1 - gamma) * self.sim[idx]
            
            mean_sim = np.mean(batch_sim)
            std_sim = np.std(batch_sim, ddof=1)
            match_label = np.ones_like(batch_sim)
            z_alpha_2 = norm.ppf(1 - self.alpha / 2)

            lower_bound = mean_sim -z_alpha_2 * std_sim
            upper_bound = mean_sim +z_alpha_2 * std_sim

            match_label[diagonal_elements > upper_bound] = 0
            match_label[diagonal_elements < lower_bound] = 2

            self.sim[idx] = batch_sim
            self.match_label[idx] = match_label
           
            loss = total_loss
        else:
            loss = total_loss
        return eeg_z, img_z, loss
    
    def training_step(self, batch, batch_idx):
        batch_size = batch['idx'].shape[0]
        eeg_z, img_z, loss = self(batch,sample_posterior=True)
        self.log('train_loss', loss, on_step=True,on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size)
        eeg_z = eeg_z/eeg_z.norm(dim=-1, keepdim=True)
        # saving traning results
        self.eeg_outputs.append(eeg_z.detach().cpu()) 
        self.img_outputs.append(img_z.detach().cpu()) 

        similarity = (eeg_z @ img_z.T)
        top_kvalues, top_k_indices = similarity.topk(5, dim=-1)
        self.all_predicted_classes.append(top_k_indices.cpu().numpy())
        label = torch.arange(0, batch_size).to(self.device)
        self.all_true_labels.extend(label.cpu().numpy())

        if batch_idx == self.trainer.num_training_batches - 1:
            all_predicted_classes = np.concatenate(self.all_predicted_classes,axis=0)
            all_true_labels = np.array(self.all_true_labels)
            top_1_predictions = all_predicted_classes[:, 0]
            top_1_correct = top_1_predictions == all_true_labels
            top_1_accuracy = sum(top_1_correct)/len(top_1_correct)
            top_k_correct = (all_predicted_classes == all_true_labels[:, np.newaxis]).any(axis=1)
            top_k_accuracy = sum(top_k_correct)/len(top_k_correct)
            self.log('train_top1_acc', top_1_accuracy, on_step=False, on_epoch=True,prog_bar=True, logger=True, sync_dist=True)
            self.log('train_top5_acc', top_k_accuracy, on_step=False, on_epoch=True,prog_bar=True, logger=True, sync_dist=True)
            self.all_predicted_classes = []
            self.all_true_labels = []

            counter = Counter(self.match_label)
            count_dict = dict(counter)
            key_mapping = {0: 'low', 1: 'medium', 2: 'high'}
            count_dict_mapped = {key_mapping[k]: v for k, v in count_dict.items()}
            self.log_dict(count_dict_mapped, on_step=False, on_epoch=True,logger=True, sync_dist=True)
            self.trainer.train_dataloader.dataset.match_label = self.match_label
        return loss

    # save training results
    def on_train_epoch_end(self):
        all_eeg = torch.cat(self.eeg_outputs, dim=0)  
        all_img = torch.cat(self.img_outputs, dim=0)
        np.save("train_eeg.npy", all_eeg.numpy()) 
        np.save("train_img.npy", all_img.numpy()) 
        self.eeg_outputs.clear()
        

    def validation_step(self, batch, batch_idx):
        batch_size = batch['idx'].shape[0]
    
        eeg_z, img_z, loss= self(batch)
        self.log('val_loss', loss, on_step=False, on_epoch=True,prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size)
        eeg_z = eeg_z/eeg_z.norm(dim=-1, keepdim=True)

        similarity = (eeg_z @ img_z.T)
        top_kvalues, top_k_indices = similarity.topk(5, dim=-1)
        self.all_predicted_classes.append(top_k_indices.cpu().numpy())
        label = torch.arange(0, batch_size).to(self.device)
        self.all_true_labels.extend(label.cpu().numpy())

        return loss
    
    def on_validation_epoch_end(self):
        all_predicted_classes = np.concatenate(self.all_predicted_classes,axis=0)
        all_true_labels = np.array(self.all_true_labels)
        top_1_predictions = all_predicted_classes[:, 0]
        top_1_correct = top_1_predictions == all_true_labels
        top_1_accuracy = sum(top_1_correct)/len(top_1_correct)
        top_k_correct = (all_predicted_classes == all_true_labels[:, np.newaxis]).any(axis=1)
        top_k_accuracy = sum(top_k_correct)/len(top_k_correct)
        self.log('val_top1_acc', top_1_accuracy, on_step=False, on_epoch=True,prog_bar=True, logger=True, sync_dist=True)
        self.log('val_top5_acc', top_k_accuracy, on_step=False, on_epoch=True,prog_bar=True, logger=True, sync_dist=True)
        self.all_predicted_classes = []
        self.all_true_labels = []

    def test_step(self,batch, batch_idx):
        batch_size = batch['idx'].shape[0]
        eeg_z, img_z, loss = self(batch)
        self.log('test_loss', loss, on_step=False, on_epoch=True,prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size)
        eeg_z = eeg_z/eeg_z.norm(dim=-1, keepdim=True)
        np.save("test_eeg.npy",eeg_z.cpu().numpy())
        np.save("test_img.npy",img_z.cpu().numpy())
        similarity = (eeg_z @ img_z.T)
        top_kvalues, top_k_indices = similarity.topk(5, dim=-1)
        self.all_predicted_classes.append(top_k_indices.cpu().numpy())
        label =  batch['label']
        self.all_true_labels.extend(label.cpu().numpy())
        return loss
        
    def on_test_epoch_end(self):
        all_predicted_classes = np.concatenate(self.all_predicted_classes,axis=0)
        all_true_labels = np.array(self.all_true_labels)
        
        top_1_predictions = all_predicted_classes[:, 0]
        top_1_correct = top_1_predictions == all_true_labels
        top_1_accuracy = sum(top_1_correct)/len(top_1_correct)
        top_k_correct = (all_predicted_classes == all_true_labels[:, np.newaxis]).any(axis=1)
        top_k_accuracy = sum(top_k_correct)/len(top_k_correct)
        self.log('test_top1_acc', top_1_accuracy, sync_dist=True)
        self.log('test_top5_acc', top_k_accuracy, sync_dist=True)
        self.all_predicted_classes = []
        self.all_true_labels = []

        avg_test_loss = self.trainer.callback_metrics['test_loss']
        return  {'test_loss': avg_test_loss.item(), 'test_top1_acc': top_1_accuracy.item(),'test_top5_acc':top_k_accuracy.item()}
        
    def configure_optimizers(self):
        optimizer = globals()[self.config['train']['optimizer']](self.parameters(), lr = self.config['train']['lr'], weight_decay=1e-4)

        return [optimizer]
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="baseline.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="eeg",
        choices=["eeg", "meg"],
        help="Choose dataset: 'eeg' or 'meg'"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="the seed (for reproducible sampling)",
    )

    parser.add_argument(
        "--subjects",
        type=str,
        default='sub-08',
        help="the subjects",
    )
    parser.add_argument(
        "--exp_setting",
        type=str,
        default='intra-subject',
        help="the exp_setting",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=50,
        help="train epoch",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="lr",
    )
    parser.add_argument(
        "--brain_backbone",
        type=str,
        help="brain_backbone",
    )
    parser.add_argument(
        "--vision_backbone",
        type=str,
        help="vision_backbone",
    )
    parser.add_argument(
        "--c",
        type=int,
        default=6,
        help="c",
    )

    opt = parser.parse_args()
    seed_everything(opt.seed)
    config = OmegaConf.load(f"{opt.config}")
    config = update_config(opt, config)
    config['data']['subjects'] = [opt.subjects]

    pretrain_map = {
        'RN50': {'pretrained': 'openai', 'resize': (224, 224), 'z_dim': 1024},
        'RN101': {'pretrained': 'openai', 'resize': (224, 224), 'z_dim': 512},
        'ViT-B-16': {'pretrained': 'laion2b_s34b_b88k', 'resize': (224, 224), 'z_dim': 512},
        'ViT-B-32': {'pretrained': 'laion2b_s34b_b79k', 'resize': (224, 224), 'z_dim': 512},
        'ViT-L-14': {'pretrained': 'laion2b_s32b_b82k', 'resize': (224, 224), 'z_dim': 768},
        'ViT-H-14': {'pretrained': 'laion2b_s32b_b79k', 'resize': (224, 224), 'z_dim': 1024},
        'ViT-g-14': {'pretrained': 'laion2b_s34b_b88k', 'resize': (224, 224), 'z_dim': 1024},
        'ViT-bigG-14': {'pretrained': 'laion2b_s39b_b160k', 'resize': (224, 224), 'z_dim': 1280}
    }

    config['z_dim'] = pretrain_map[opt.vision_backbone]['z_dim']
    print(config)

    os.makedirs(config['save_dir'],exist_ok=True)
    logger = TensorBoardLogger(config['save_dir'], name=config['name'], version=f"{'_'.join(config['data']['subjects'])}_seed{config['seed']}")
    os.makedirs(logger.log_dir,exist_ok=True)
    shutil.copy(opt.config, os.path.join(logger.log_dir,opt.config.rsplit('/',1)[-1]))

    train_loader, val_loader, test_loader = load_eeg_data(config) if config['dataset'] == 'eeg' else load_meg_data(config)

    print(f"train num: {len(train_loader.dataset)},val num: {len(val_loader.dataset)}, test num: {len(test_loader.dataset)}")
    pl_model = load_model(config, train_loader, test_loader)

    checkpoint_callback = ModelCheckpoint(save_last=True)

    if config['exp_setting'] == 'inter-subject':
        early_stop_callback = EarlyStopping(
            monitor='val_top1_acc',
            min_delta=0.001,     
            patience=5, 
            verbose=False,
            mode='max' 
        )
    else:
        early_stop_callback = EarlyStopping(
            monitor='train_loss',
            min_delta=0.001,
            patience=5,
            verbose=False,  
            mode='min' 
        )

    trainer = Trainer(log_every_n_steps=10, strategy=DDPStrategy(find_unused_parameters=False),callbacks=[early_stop_callback, checkpoint_callback],max_epochs=config['train']['epoch'], devices=[device],accelerator='cuda',logger=logger)
    print(trainer.logger.log_dir)

    ckpt_path = "/home/yueming/github/Uncertainty-aware-Blur-Prior/exp/intra-subject_ubp_EEGProjectLayer_RN50/sub-10_seed0/checkpoints/last-v10.ckpt"
    trainer.fit(pl_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    if config['exp_setting'] == 'inter-subject':
        test_results = trainer.test(ckpt_path='best', dataloaders=test_loader)
    else:
        test_results = trainer.test(ckpt_path=ckpt_path, dataloaders=test_loader)

    with open(os.path.join(logger.log_dir,'test_results.json'), 'w') as f:
        json.dump(test_results, f, indent=4)

if __name__=="__main__":
    main()