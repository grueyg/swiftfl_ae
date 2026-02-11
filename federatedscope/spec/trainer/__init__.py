from federatedscope.register import register_trainer
from federatedscope.spec.trainer.client_trainer import SpecTrainer
from federatedscope.spec.trainer.client_trainer import SpecNLPTrainer
from federatedscope.spec.trainer.server_updater import SpecUpdater
from federatedscope.spec.exp.fednova.fednova_trainer import FedNovaTrainer
from federatedscope.spec.exp.scaffold.scaffold_trainer import ScaffoldTrainer
from federatedscope.spec.exp.fjord.fjord_trainer import FjORDTrainer, FjORDNLPTrainer

def call_spec_trainer(trainer_type):
    trainer_type = trainer_type.lower()
    if trainer_type == 'spectrainer':
        trainer_builder = SpecTrainer
    elif trainer_type == 'specnlptrainer':
        trainer_builder = SpecNLPTrainer
    elif trainer_type == 'scaffoldtrainer':
        trainer_builder = ScaffoldTrainer
    elif trainer_type == 'fjordtrainer':
        trainer_builder = FjORDTrainer
    elif trainer_type == 'fjordnlptrainer':
        trainer_builder = FjORDNLPTrainer
    else:
        return None
    return trainer_builder

register_trainer('SpecTrainer', call_spec_trainer)

__all__ = ['SpecTrainer', 'SpecNLPTrainer', 'SpecUpdater', 'FedNovaTrainer', 'ScaffoldTrainer', 'FjORDTrainer']
