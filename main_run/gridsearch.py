import itertools
import numpy as np
from main_run.train import Exp_Main
#from main_run.nbeats import Exp_NBeats
from main_run.nbeats_mod2 import Exp_NBeats_m2
from main_run.nbeats_bcast import Exp_Bcast


TYPES = {0: 'Original',
         1: 'Gaussian Noise',
         2: 'Freq-Mask',
         3: 'Freq-Mix',
         4: 'Wave-Mask',
         5: 'Wave-Mix',
         6: 'Wave-MixUp',
         7: 'StAug',
         8: 'NBeats (basic)',
         9: 'NBeats (pretrained)'}

class GridSearch(object):
  
  def __init__(self, args):
      self.args = args

  def makegrid(self, pars_dict):

    keys=pars_dict.keys()
    combinations = itertools.product(*pars_dict.values())
    ds=[dict(zip(keys,cc)) for cc in combinations]
    return ds

  def grid_search(self, params_dicts, attrs, setting):
    self.mse_per_parameter = []
    self.params = params_dicts[self.args.aug_type - 1]
    self.args_attrs = attrs[self.args.aug_type - 1]
    self.hyperparameters = self.makegrid(self.params)

    for hp in self.hyperparameters:
      for attr_name in self.args_attrs:
        setattr(self.args, attr_name, hp[attr_name])

      if self.args.nbeats == 1: 
        exp = Exp_NBeats(self.args)
      elif self.args.nbeats == 2: 
        exp1 = Exp_Bcast(self.args)
        exp2 = Exp_NBeats_m2(self.args)
      else: 
        exp = Exp_Main(self.args)  

      if self.args.nbeats == 2:
        fcasts, bcasts = exp1.train(setting[0])
        _, mse = exp2.train(setting[1], fcasts, bcasts)
        mse_test, _, _ = exp2.test(setting[1])
        print(f"For {hp}, the MSE: {mse}")
        self.mse_per_parameter.append(mse)
      else: 
        _, mse = exp.train(setting)
        mse_test, _, _ = exp.test(setting, test = 1)
        print(f"For {hp}, the MSE: {mse}")
        self.mse_per_parameter.append(mse)

      f = open("grid_search-" + TYPES[self.args.aug_type] + self.args.des + self.args.data + ".txt", 'a')
      f.write(" \n")
      f.write('For hp {}, mse:{}'.format(hp,mse))
    f.write(" \n") 
    f.write("-------------------------------------------------- {} --------------------------------------------------------------------------------------".format(TYPES[self.args.aug_type]))
    f.write(" \n")
    f.write('The best hyperparameter:{} with loss {}'.format(self.hyperparameters[self.mse_per_parameter.index(min(self.mse_per_parameter))], min(self.mse_per_parameter)))
    f.write(" \n")
    f.write("-------------------------------------------------- {} --------------------------------------------------------------------------------------".format(TYPES[self.args.aug_type]))

    f.write('The second best hyperparameter:{} with loss {}'.format(self.hyperparameters[self.mse_per_parameter.index(sorted(self.mse_per_parameter)[1])], sorted(self.mse_per_parameter)[1]))
    f.write(" \n")
    f.write("-------------------------------------------------- {} --------------------------------------------------------------------------------------".format(TYPES[self.args.aug_type]))
    f.write(" \n")
    f.write('The third best hyperparameter:{} with loss {}'.format(self.hyperparameters[self.mse_per_parameter.index(sorted(self.mse_per_parameter)[2])], sorted(self.mse_per_parameter)[2]))
    f.write(" \n")
    f.close()
    return self.hyperparameters[self.mse_per_parameter.index(min(self.mse_per_parameter))]
