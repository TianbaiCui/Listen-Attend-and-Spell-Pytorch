import yaml
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from model.las_model import LAS
from datamodule_torchaudio import DataModule
from trainmodule import LASTrainModule

with open("./config/las_libri_config.yaml", "rb") as f:
    conf = yaml.load(f, Loader=yaml.FullLoader)

conf_path = "./config/las_libri_config.yaml"
data_path =  "/disk2/data/cache/"

# Parameters loading
print()
print('Experiment :',conf['meta_variable']['experiment_name'])
total_steps = conf['training_parameter']['total_steps']

listener_model_path = conf['meta_variable']['checkpoint_dir']+conf['meta_variable']['experiment_name']+'.listener'
speller_model_path = conf['meta_variable']['checkpoint_dir']+conf['meta_variable']['experiment_name']+'.speller'
verbose_step = conf['training_parameter']['verbose_step']
valid_step = conf['training_parameter']['valid_step']
tf_rate_upperbound = conf['training_parameter']['tf_rate_upperbound']
tf_rate_lowerbound = conf['training_parameter']['tf_rate_lowerbound']
tf_decay_step = conf['training_parameter']['tf_decay_step']
seed = conf['training_parameter']['seed']

datamodule = DataModule(data_path, conf_path)
model = LAS(tf_rate=1, **conf['model_parameter'], **conf['training_parameter'])

model_train = LASTrainModule(
    model=model, 
    max_label_len=conf['model_parameter']['max_label_len'], 
    label_smoothing=conf['model_parameter']['label_smoothing'],
    tf_rate_lower=tf_rate_lowerbound,
    tf_rate_upper=tf_rate_upperbound, 
    tf_decay_step=tf_decay_step, 
    fn_int_to_text = datamodule.int_to_text
    )

wandb_logger = WandbLogger()

checkpoint_callback = ModelCheckpoint(
    dirpath="./model_checkpoints",
    filename="model_checkpoint",
    verbose=True,
    save_top_k=1,
    save_weights_only=True,
    monitor="val_loss",
    mode="min",
)

early_stop_callback = EarlyStopping(
    monitor="val_loss", patience=3, verbose=True, mode="min"
)

trainer = Trainer(
    progress_bar_refresh_rate=100,
    max_epochs=100,
    gpus=[0],
    # auto_select_gpus=True,
    # accelerator="ddp",
    checkpoint_callback=checkpoint_callback,
    callbacks=[early_stop_callback],
    # fast_dev_run=True,
    logger=wandb_logger,
)

trainer.fit(model=model_train, datamodule=datamodule)