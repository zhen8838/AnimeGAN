import tensorflow as tf
from models.gannet import dcgan_mnist, pix2pix_facde, animenet
from tools.custom import StepLR, CosineLR, ScheduleLR
from tools.animegan import AnimeGanHelper, AnimeGanInitLoop, AnimeGanLoop
from tools.training_engine import BaseTrainingLoop
from yaml import safe_dump


class dict2obj(object):

  def __init__(self, dicts):
    """ convert dict to object , NOTE the `**kwargs` will not be convert 

        Parameters
        ----------
        object : [type]

        dicts : dict
            dict
        """
    for name, value in dicts.items():
      if isinstance(value, (list, tuple)):
        setattr(self, name,
                [dict2obj(x) if isinstance(x, dict) else x for x in value])
      else:
        if 'kwarg' in name:
          setattr(self, name, value if value else dict())
        else:
          if isinstance(value, dict):
            setattr(self, name, dict2obj(value))
          else:
            setattr(self, name, value)

  def keys(self):
    return self.__dict__.keys()

  def items(self):
    return self.__dict__.items()

  def values(self):
    return self.__dict__.values()


ArgDict = {
    'mode': 'train',

    # MODEL
    'model': {
        'name': 'yolo',
        'helper': 'YOLOHelper',
        'helper_kwarg': {
            'image_ann': 'data/voc_img_ann.npy',
            'class_num': 20,
            'anchors': 'data/voc_anchor.npy',
            'in_hw': [224, 320],
            'out_hw': [[7, 10], [14, 20]],
            'validation_split': 0.3,  # vaildation_split
        },
        'network': 'yolo_mbv2_k210',
        'network_kwarg': {
            'input_shape': [224, 320, 3],
            'anchor_num': 3,
            'class_num': 20,
            'alpha': 0.75  # depth_multiplier
        },
        'loss': 'YOLOLoss',
        'loss_kwarg': {
            'obj_thresh': 0.7,
            'iou_thresh': 0.5,
            'obj_weight': 1,
            'noobj_weight': 1,
            'wh_weight': 1,
        }
    },
    'train': {
        'jit': True,
        'augmenter': False,
        'batch_size': 16,
        'pre_ckpt': None,
        'rand_seed': 10101,
        'epochs': 10,
        'log_dir': 'log',
        'sub_log_dir': None,
        'debug': False,
        'verbose': 1,
        'vali_step_factor': 0.5,
        'optimizer': 'RAdam',
        'optimizer_kwarg': {
            'lr': 0.001,  # init_learning_rate
            'beta_1': 0.9,
            'beta_2': 0.999,
            'epsilon': None,
            'decay': 0.  # learning_rate_decay_factor
        },
        'Lookahead': True,
        'Lookahead_kwarg': {
            'k': 5,
            'alpha': 0.5,
        },
        'earlystop': True,
        'earlystop_kwarg': {
            'monitor': 'val_loss',
            'min_delta': 0,
            'patience': 4,
            'verbose': 0,
            'mode': 'auto',
            'baseline': None,
            'restore_best_weights': False,
        },
        'modelcheckpoint': True,
        'modelcheckpoint_kwarg': {
            'monitor': 'val_loss',
            'verbose': 0,
            'save_best_only': True,
            'save_weights_only': False,
            'mode': 'auto',
            'save_freq': 'epoch',
            'load_weights_on_restart': False,
        },
    },
    'prune': {
        'is_prune': False,
        'init_sparsity': 0.5,  # prune initial sparsity range = [0 ~ 1]
        'final_sparsity': 0.9,  # prune final sparsity range = [0 ~ 1]
        'end_epoch': 5,  # prune epochs NOTE: must < train epochs
        'frequency': 100,  # how many steps for prune once
    },
    'inference': {
        'infer_fn': 'yolo_infer',
        'infer_fn_kwarg': {
            'obj_thresh': .7,
            'iou_thresh': .3
        },
    },
    'evaluate': {
        'eval_fn': 'yolo_eval',
        'eval_fn_kwarg': {
            'det_obj_thresh': 0.1,
            'det_iou_thresh': 0.3,
            'mAp_iou_thresh': 0.3
        },
    }
}

helper_register = {
    'AnimeGanHelper': AnimeGanHelper,
}

network_register = {
    'dcgan_mnist': dcgan_mnist,
    'pix2pix_facde': pix2pix_facde,
    'animenet': animenet,
}

loss_register = {}

callback_register = {
    'EarlyStopping': tf.keras.callbacks.EarlyStopping,
    'ModelCheckpoint': tf.keras.callbacks.ModelCheckpoint,
    'TerminateOnNaN': tf.keras.callbacks.TerminateOnNaN,
    'StepLR': StepLR,
    'CosineLR': CosineLR,
    'ScheduleLR': ScheduleLR,
}

optimizer_register = {
    'Adam': tf.keras.optimizers.Adam,
    'SGD': tf.keras.optimizers.SGD,
    'RMSprop': tf.keras.optimizers.RMSprop,
    'Adamax': tf.keras.optimizers.Adamax,
    'Nadam': tf.keras.optimizers.Nadam,
    'Ftrl': tf.keras.optimizers.Ftrl,
}

infer_register = {}

eval_register = {}

trainloop_register = {
    'BaseTrainingLoop': BaseTrainingLoop,
    'AnimeGanInitLoop': AnimeGanInitLoop,
    'AnimeGanLoop': AnimeGanLoop,
}

if __name__ == "__main__":
  with open('config/default.yml', 'w') as f:
    safe_dump(ArgDict, f, sort_keys=False)
