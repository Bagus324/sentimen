from utils.preprocessor import Preprocessor
from utils.trainer import MultiClassTrainer

from transformers import BertConfig
import pytorch_lightning as pl


if __name__ =="__main__":
    dm = Preprocessor()

    config = BertConfig()


    model = MultiClassTrainer(
        lr = 1e-5,
        bert_config = config,
        dropout=0.3
    )
    trainer = pl.Trainer(gpus = 1, 
                        max_epochs = 20, 
                        default_root_dir = "./checkpoints/class",
                        )

    trainer.fit(model, datamodule = dm)

    trainer.predict(model = model, datamodule = dm)