from copy import deepcopy as copy_deep

class IvyXGBoostParameters:
   """
   A container of XGBoost booster parameters.
   """
   def __init__(self):
      self.params = dict()
      # From https://xgboost.readthedocs.io/en/stable/parameter.html
      ## General parameters
      self.params['booster'] = "gbtree" # Only option supported at the moment.
      self.params['nthread'] = 1 # Number of parallel threads used to run XGBoost.
      ## Parameters for tree booster
      self.params['eta'] = 0.07 # Step size shrinkage used in update to prevents overfitting.
      self.params['num_round'] = 500 # Number of rounds for boosting. Remember to increase this value for small 'eta'.
      self.params['gamma'] = 2.0 # Minimum loss reduction required to make a further partition on a leaf node of the tree.
      self.params['max_depth'] = 5 # Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit.
      self.params['min_child_weight'] = 1.0 # Minimum sum of instance weight (hessian) needed in a child to accept partitioning.
      self.params['max_delta_step'] = 0.0 # Maximum delta step we allow each leaf output to be. If the value is set to 0, it means there is no constraint.
      self.params['subsample'] = 0.6 # Subsample ratio of the training instances to randomly sample the training data prior to growing trees in order to prevent overfitting.
      self.params['sampling_method'] = "uniform" # The method to use to sample the training instances. See above link for the different options.
      self.params['colsample_bytree'] = 1.0 # Subsample ratio of columns when constructing each tree.
      self.params['colsample_bylevel'] = 1.0 # Subsample ratio of columns for each level.
      self.params['colsample_bynode'] = 1.0 # Subsample ratio of columns for each node (split).
      self.params['alpha'] = 8.0 # L1 regularization term on weights. Increasing this value will make model more conservative.
      self.params['lambda'] = 1.0 # L2 regularization term on weights. Increasing this value will make model more conservative.
      self.params['tree_method'] = "auto" # The tree construction algorithm used in XGBoost. See above link for the details.
      #self.params['scale_pos_weight'] = 1.0
      # Control the balance of positive and negative weights, useful for unbalanced classes.
      # We actually rebalance weights on our own and do not use this feature.
      # In multi-class training, it gets more complicated than just a simple factor.
      ## Learning task parameters
      #self.params['num_class'] = 2
      #self.params['objective'] = "binary:logistic"
      # Options for 'objective' are
      # - 'binary:logistic' for logistic regression for binary classification with output as a probability.
      # - 'multi:softprob' for multiclass classification using the softmax objective with output as the probability of each class. Needs 'num_class' to be set as well, which we do internally.
      # See the above link for more options ans explanations.
      # We sety both 'objective' and 'num_class' ourselves.
      self.params['silent'] = 1
      self.params['eval_metric'] = "logloss"
      # Evaluation metrics for validation data; switched to mlogloss when objective=multi:softprob.
      # Cold also be set to 'auc'.


   def setParameters(self, **kwargs):
      """
      Call to set parameters that are already defined.
      This function checks if they are defined, and throws a runtime error otherwise.
      """
      for key in kwargs:
         if key not in self.params.keys():
            raise RuntimeError("IvyXGBoostParameters::setParameters: The argument '{}' is not part of the set of parameters.".format(key))
         else:
            target_type = type(self.params[key])
            if target_type != type(kwargs[key]):
               raise RuntimeError("IvyXGBoostParameters::setParameters: The type of the argument '{}' should be {}.".format(key, target_type))
            self.params[key] = kwargs[key]


   def defineParameters(self, **kwargs):
      """
      Call to define parameters other than what is already defined through the constructor and initialize them.
      This function does not check whether a parameter is already defined, so it can also be called as an unchecked version of IvyXGBoostParameters::setParameters.
      However, it checks the types of parameters that are already defined. The call throws a runtime error if the types are inconsistent.
      """
      for key in kwargs:
         if key in self.params.keys():
            target_type = type(self.params[key])
            if target_type != type(kwargs[key]):
               raise RuntimeError("IvyXGBoostParameters::defineParameters: The type of the argument '{}' should be {}.".format(key, target_type))
         self.params[key] = kwargs[key]


   def getParameters(self):
      """
      Call to get the set of parameters in order to use them in training.
      Note that the returned dictionary is a deep copy, so the user should not expact the original class member 'params' to be returned.
      The user is not supposed to have access to that class member other than through the setParameters and defineParameters functions!
      """
      return copy_deep(self.params)
