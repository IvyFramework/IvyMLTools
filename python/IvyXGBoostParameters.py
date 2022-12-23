class IvyXGBoostParameters:
   def __init__(self):
      self.params = dict()
      self.params['objective'] = 'binary:logistic'
      self.params['eta'] = 0.07
      self.params['max_depth'] = 5
      self.params['silent'] = 1
      self.params['nthread'] = 1
      self.params['eval_metric'] = "auc"
      self.params['subsample'] = 0.6
      self.params['alpha'] = 8.0
      self.params['gamma'] = 2.0
      self.params['lambda'] = 1.0
      self.params['min_child_weight'] = 1.0
      self.params['colsample_bytree'] = 1.0
      self.params["scale_pos_weight"] = 1.0
      self.num_round = 500


   def setParameters(self, **kwargs):
      for key in kwargs:
         if key=="num_round":
            target_type = type(self.num_round)
            if target_type != type(kwargs[key]):
               raise RuntimeError("IvyXGBoostParameters::setParameters: The type of the argument 'num_round' should be {}.".format(key, target_type))
            self.num_round = kwargs[key]
         elif key not in self.params.keys():
            raise RuntimeError("IvyXGBoostParameters::setParameters: The argument '{}' is not part of the set of parameters.".format(key))
         else:
            target_type = type(self.params[key])
            if target_type != type(kwargs[key]):
               raise RuntimeError("IvyXGBoostParameters::setParameters: The type of the argument '{}' should be {}.".format(key, target_type))
            self.params[key] = kwargs[key]


   def getXGBoostParameters(self):
      return self.params


   def getXGBoostNumRounds(self):
      return self.num_rounds
