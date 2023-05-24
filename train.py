import os
from pprint import pprint
from configs.config_mimic import parser
from dataset.data_module import DataModule
from lightning_tools.callbacks import add_callbacks
from models.swinllama import SwinLLama
import pytorch_lightning as pl
 
 
def train(args):
    dm = DataModule(args)
    callbacks = add_callbacks(args)
    trainer = pl.Trainer.from_argparse_args(
        args, callbacks=callbacks["callbacks"], logger=callbacks["loggers"]
    )
    if args.ckpt_file is not None:
        model = SwinLLama.load_from_checkpoint(args.ckpt_file, strict=False)
    else:
        model = SwinLLama(args)
    trainer.fit(model, datamodule=dm)


def main():
    args = parser.parse_args()
    os.makedirs(args.savedmodel_path, exist_ok=True)
    pprint(vars(args))
    train(args)


if __name__ == '__main__':
    main()