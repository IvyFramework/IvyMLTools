import uproot
import numpy as np
from sklearn.model_selection import train_test_split


class IvyXGBoostDataInput:
   def __init__(self, feature_names, class_branch_name = None, missing_value_default = -999.):
      """
      IvyXGBoostDataInput constructor:
      - file_name: Input ROOT file name
      - tree_name: Name of TTree in the input file
      - feature_names: A list of input 'features' that will be read from the input TTree.
      - class_branch_name: Name of the branch that indicates the 'class' of the data entry. Default: None.
      - missing_value_default: Missing value indicator. Default: -999 (float).
      """
      self.features = feature_names
      if type(self.features) is str:
         self.features = self.features.split(", ")
      if len(self.features)==0:
         raise RuntimeError("There should be at least one feature name.")
      self.class_branch = class_branch_name
      if self.class_branch is not None and type(self.class_branch) is not str:
         raise RuntimeError("The class branch should be specified as a string.")
      if self.class_branch in self.features:
         self.features.remove(self.class_branch)
      self.missing_value_default = np.float32(missing_value_default)
      self.data_train = None
      self.data_test = None
      self.data_control = None


   def add_data(self, features_data, weights, class_values, train_fraction, control_fraction=None, shuffle=None):
      """
      Add data from an existing set of lists.
      - features_data: 2D numpy array of floats for the feature values arranged as [rows=entries][columns=features]
      - weights: 1D numpy array of floats, or a single floating value, for the weight values
      - class_values: 1D numpy array of ints, or a single integer value, for the class values
      - train_fraction (fT): Fraction of data used for training
      - control_fraction (fC): Fraction of data not used in any training or evaluation (default = None, i.e., inactive)
      - shuffle: If True, the data entries for the training, evaluation, and control (if enabled) samples will be split randomly.

      The fractions of the training, evaluation, and control samples are calculated as (1-fC)*fT, (1-fC)*(1-fT), and fC, respectively.

      If there is more than one class in the added data set, the user must set 'shuffle' to True or False; it cannot be kept as None.
      Otherwise, the behavior for shuffle=None is the same as that for shuffle=False.
      """
      if isinstance(class_values, int) or isinstance(class_values, np.integer):
         class_values = np.full(features_data.shape[0], class_values, dtype=np.int32)
      if isinstance(weights, float) or isinstance(weights, np.floating):
         weights = np.full(features_data.shape[0], weights, dtype=np.float32)

      class_types = np.unique(class_values)
      nClasses = class_types.size
      if nClasses > 1 and shuffle is None:
         raise RuntimeError("IvyXGBoostDataInput::add_data: The option 'shuffle' is None, but there is more than one class in the added data. To ensure this behavior is intended, this function requires shuffle to be set in this special case.")
      if shuffle is None:
         shuffle = False

      if features_data.shape[0]!=weights.shape[0] or features_data.shape[0]!=class_values.shape[0]:
         raise RuntimeError("IvyXGBoostDataInput::add_data: The number of rows in features, weights, and class values data should be the same.")

      feat_data=[None,None,None]
      wgt_data=[None,None,None]
      class_data=[None,None,None]
      if control_fraction is not None:
         feat_data[2], feat_data[1], wgt_data[2], wgt_data[1], class_data[2], class_data[1] = train_test_split(
            features_data, weights, class_values,
            train_size = control_fraction,
            random_state = 12345,
            shuffle = shuffle
         )
         features_data = feat_data[1]
         weights = wgt_data[1]
         class_values = class_data[1]
      feat_data[0], feat_data[1], wgt_data[0], wgt_data[1], class_data[0], class_data[1] = train_test_split(
         features_data, weights, class_values,
         train_size = train_fraction,
         random_state = 12345,
         shuffle = shuffle
      )

      data_train = [feat_data[0], wgt_data[0], class_data[0]]
      data_test = [feat_data[1], wgt_data[1], class_data[1]]
      data_control = None
      if control_fraction is not None:
         data_control = [feat_data[2], wgt_data[2], class_data[2]]
      if self.data_train is None:
         self.data_train = data_train
         self.data_test = data_test
      else:
         for idx in range(0, 3):
            if self.data_train[idx].ndim == 1:
               self.data_train[idx] = np.concatenate([self.data_train[idx], data_train[idx]])
               self.data_test[idx] = np.concatenate([self.data_test[idx], data_test[idx]])
            else:
               self.data_train[idx] = np.row_stack([self.data_train[idx], data_train[idx]])
               self.data_test[idx] = np.row_stack([self.data_test[idx], data_test[idx]])
      if data_control is not None:
         if self.data_control is None:
            self.data_control = data_control
         else:
            for idx in range(0, 3):
               if self.data_control[idx].ndim == 1:
                  self.data_control[idx] = np.concatenate([self.data_control[idx], data_control[idx]])
               else:
                  self.data_control[idx] = np.row_stack([self.data_control[idx], data_control[idx]])


   def load_input(self, file_name, tree_name, train_fraction, control_fraction=None, shuffle=None, weight_name=None, class_type=None):
      """
      Loads a ROOT file with a TTree in it.
      - file_name: Input ROOT file name
      - tree_name: Name of TTree in the input file
      - train_fraction (fT): Fraction of data used for training
      - control_fraction (fC): Fraction of data not used in any training or evaluation (default = None, i.e., inactive)
      - shuffle: If True, the data entries for the training, evaluation, and control (if enabled) samples will be split randomly.
      - weight_name: Name of the branch that contains weights. Optional.
      - class_type: If an integer value is given, the value of self.class_branch will be set to this one.

      For the descripton of how fT and fC are used, please see the help for IvyXGBoostDataInput::add_data.
      """
      output_vars = [ v for v in self.features ]
      output_vars.append("weight")
      output_vars.append("data_class")

      assign_class = (class_type is not None)
      is_weighted = (weight_name is not None)

      finput = uproot.open(file_name)
      tin = finput[tree_name]
      keylist = tin.keys()

      input_vars = [ v for v in self.features ]
      if is_weighted:
         if weight_name in keylist:
            input_vars.append(weight_name)
         else:
            raise RuntimeError("IvyXGBoostDataInput::load_input: The weight branch {} does not exist in the input tree.".format(weight_name))
      if not assign_class:
         if self.class_branch is None:
            raise RuntimeError("IvyXGBoostDataInput::load_input: Because IvyXGBoostDataInput was constructed with no inherent class branch name, the class type needs to be specified.")
         if self.class_branch in keylist:
            input_vars.append(self.class_branch)
         else:
            raise RuntimeError("IvyXGBoostDataInput::load_input: Class name {} is not in the list of branches.".format(self.class_branch))
      elif type(class_type) is not int:
         raise RuntimeError("IvyXGBoostDataInput::load_input: Class type should be specified as an integer.")
      arrs = tin.arrays(input_vars)
      nEntries = arrs[input_vars[0]].shape[0]

      weights = None
      if is_weighted:
         weights = arrs[weight_name].astype(np.float32)
      else:
         weights = np.ones(nEntries, dtype=np.float32)

      class_values = None
      if not assign_class:
         class_values = arrs[self.class_branch].astype(np.int32)
      else:
         class_values = np.full(nEntries, class_type, dtype=np.int32)

      feat_data = np.column_stack([ arrs[v] for v in self.features ])
      self.add_data(feat_data, weights, class_values, train_fraction, control_fraction, shuffle)


   def class_types(self):
      """
      Returns the ordered set of class values.
      """
      res = None
      if self.data_train is not None:
         clm = np.unique(self.data_train[2])
         cll = [ clm[i] for i in range(clm.shape[0]) ]
         res = cll
      if res is None:
         raise RuntimeError("IvyXGBoostDataInput::class_types: This function should only be called after assigning a training data set.")
      return res
