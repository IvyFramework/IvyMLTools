import uproot
import numpy as np
from sklearn.model_selection import train_test_split


class IvyXGBoostDataInput:
   def __init__(self, feature_names, class_branch_name = None):
      """
      IvyXGBoostDataInput constructor:
      - file_name: Input ROOT file name
      - tree_name: Name of TTree in the input file
      - feature_names: A list of input 'features' that will be read from the input TTree.
      - class_branch_name: Name of the branch that indicates the 'class' of the event. Default: None.
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
      self.data_train = None
      self.data_test = None


   def load_input(self, file_name, tree_name, train_fraction=0.5, weight_name = None, class_type = None):
      """
      Loads a ROOT file with a TTree in it.
      - file_name: Input ROOT file name
      - tree_name: Name of TTree in the input file
      - train_fraction: Fraction of data used for training (default = 0.5)
      - weight_name: Name of the branch that contains weights. Optional.
      - class_type: If an integer value is given, the value of self.class_branch will be set to this one.
      """
      output_vars = [ v for v in self.features ]
      output_vars.append("weight")
      output_vars.append("event_class")

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
      feat_train, feat_test, wgt_train, wgt_test, class_train, class_test = train_test_split(
         feat_data, weights, class_values,
         train_size = train_fraction,
         random_state = 12345,
         shuffle = False
      )

      data_train = [feat_train, wgt_train, class_train]
      data_test = [feat_test, wgt_test, class_test]
      if self.data_train is None:
         self.data_train = data_train
         self.data_test = data_test
      else:
         for idx in range(0, 3):
            self.data_train[idx] = np.row_stack([self.data_train[idx], data_train[idx]])
            self.data_test[idx] = np.row_stack([self.data_test[idx], data_test[idx]])
