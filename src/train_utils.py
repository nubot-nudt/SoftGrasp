import os
from datetime import datetime
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from datetime import datetime
import numpy as np
import optuna

def save_config(args):
    config_name = os.path.basename(args.config).split(".yaml")[0]
    now = datetime.now()
    dt = now.strftime("%m%d%Y")
    exp_dir = os.path.join("policy","exp" + dt + args.task)
    # exp_dir = os.path.join("exp" + dt + args.task, config_name)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    with open(os.path.join(exp_dir, "conf.yaml"), "w") as outfile:
        yaml.safe_dump(vars(args), outfile)
    return exp_dir

def start_training(args, exp_dir, pl_module,monitor="val/loss"):
    start_time = datetime.now()  
    jobid = np.random.randint(0, 1000)
    jobid = os.environ.get("SLURM_JOB_ID", 0)
    exp_time = datetime.now().strftime("%m-%d-%H:%M:%S")
    if args.train:

        study = optuna.create_study(direction='minimize')
        objective = lambda trial: trial.suggest_float("lambda_param", 0.1, 1.0)  
        study.optimize(objective, n_trials=args.num_trials)
        best_trial = study.best_trial
        lambda_param = best_trial.params["lambda_param"]

        pl_module.set_lambda_param(lambda_param)

        checkpoint = ModelCheckpoint(
            dirpath=os.path.join(exp_dir, "checkpoints", args.task),
            filename=exp_time+ args.task + "-{epoch}-{step}",
            save_top_k = 1,  
            save_last=True,
            monitor=monitor,
            # mode="max",
            # ``'val_acc'``, this should be ``'max'``
            # ``'val_loss'`` this should be ``'min'``
            mode="min",
        )

        logger = TensorBoardLogger(
            save_dir=exp_dir, version=exp_time + args.task, name="lightning_logs"
        )
        trainer = Trainer(
            max_epochs=args.epochs,
            callbacks=[checkpoint],
            default_root_dir=exp_dir,
            # gpus=-1,
            strategy="auto",
            check_val_every_n_epoch=1,
            log_every_n_steps=1,
            logger=logger,
            precision =16,
            gradient_clip_val=args.gradient_clip_val 
        )
        trainer.fit(
            pl_module,
            # ckpt_path='policy/exp06062024j_f_vf_share/checkpoints/j_f_vf_share/06-06-16:30:20j_f_vf_share-epoch=10-step=16005.ckpt'
            ckpt_path=None
            if args.resume is None
            else os.path.join(os.getcwd(), args.resume),
        )

        end_time = datetime.now()   
        total_time = end_time - start_time  
        print(f"Training started at {start_time} and ended at {end_time}")
        print(f"Total training time: {total_time}")
        print("best_model", checkpoint.best_model_path)
        print("best_model_score", checkpoint.best_model_score)
        args.training_end_time = end_time.strftime("%Y-%m-%d %H:%M:%S")
        args.total_training_time = str(total_time)
        args.best_model_path = checkpoint.best_model_path
        args.best_model_score = float(checkpoint.best_model_score)

        with open(os.path.join(exp_dir, "conf.yaml"), "w") as outfile:
            yaml.safe_dump(vars(args), outfile)

    else:
        logger = TensorBoardLogger(
            save_dir=exp_dir, version=exp_time + "test_" + args.task, name="lightning_logs"
        )
        trainer = Trainer(
                precision=16,
                logger=logger
            )
        trainer.test(model=pl_module,ckpt_path=args.ckpt)
