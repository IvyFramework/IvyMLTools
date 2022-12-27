import os
import sys
import numpy as np
import xgboost as xgb
from IvyXGBoostDataInput import IvyXGBoostDataInput
from IvyXGBoostParameters import IvyXGBoostParameters


class IvyXGBoostTrainer:
   def __init__(self):
      self.booster = None
      self.prediction_train = None
      self.prediction_test = None
      self.prediction_control = None


   def train(self, xgb_input, xgb_params, early_stopping_rounds=None, scale_weights=True, save_predictions=False):
      data_train = xgb_input.data_train
      data_test = xgb_input.data_test
      data_control = xgb_input.data_control
      hasControlData = (data_control is not None)

      params = xgb_params.getParameters()

      classes = np.unique(data_train[2])
      nClasses = classes.size
      if nClasses==1:
         raise RuntimeError("IvyXGBoostTrainer::train: Cannot train with only one class.")
      else:
         print("IvyXGBoostTrainer::train: {} classes were identified.".format(nClasses))
         if nClasses>2:
            params['num_class'] = nClasses
            if 'objective' not in params.keys():
               params['objective'] = "multi:softprob"
            if params['eval_metric'] == "logloss":
               params['eval_metric'] = "mlogloss"
         else:
            if 'objective' not in params.keys():
               params['objective'] = "binary:logistic"
            if params['eval_metric'] == "mlogloss":
               params['eval_metric'] = "logloss"
         print("- objective = {}".format(params['objective']))
         print("- eval_metric = {}".format(params['eval_metric']))

      wgts_train = np.abs(data_train[1])
      wgts_test = np.abs(data_test[1])
      wgts_control = np.abs(data_control[1]) if hasControlData else None
      if scale_weights:
         navg_train = np.float32(wgts_train.size)/np.float32(nClasses)
         navg_test = np.float32(wgts_test.size)/np.float32(nClasses)
         navg_control = np.float32(wgts_control.size)/np.float32(nClasses) if hasControlData else None
         for cls in classes:
            sum_train = wgts_train[data_train[2]==cls].sum()
            sum_test = wgts_test[data_test[2]==cls].sum()
            sum_control = wgts_control[data_control[2]==cls].sum() if hasControlData else None
            wgts_train[data_train[2]==cls] *= navg_train / sum_train
            wgts_test[data_test[2]==cls] *= navg_test / sum_test
            if hasControlData:
               wgts_control[data_control[2]==cls] *= navg_control / sum_control

      dtrain = xgb.DMatrix( data_train[0], label=data_train[2], weight=wgts_train, feature_names=xgb_input.features )
      dtest = xgb.DMatrix( data_test[0], label=data_test[2], weight=wgts_test, feature_names=xgb_input.features )
      dcontrol = xgb.DMatrix( data_control[0], label=data_control[2], weight=wgts_control, feature_names=xgb_input.features ) if hasControlData else None
      eval_list = [(dtrain,'train'), (dtest,'eval')]
      self.booster = xgb.train(params, dtrain, params['num_round'], eval_list, early_stopping_rounds=early_stopping_rounds)

      if save_predictions:
         print("IvyXGBoostTrainer::train: Saving the predictions...")
         self.prediction_train = self.booster.predict(dtrain)
         self.prediction_test = self.booster.predict(dtest)
         if hasControlData:
            self.prediction_control = self.booster.predict(dcontrol)
         else:
            self.prediction_control = None


   def save_model(self, fname):
      if self.booster is not None:
         if fname.endswith(".json"):
            self.booster.dump_model(fname)
         else:
            self.booster.save_model(fname)
