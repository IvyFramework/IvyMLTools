import random
import numpy as np
import json
from IvyXGBoostParameters import IvyXGBoostParameters
from IvyXGBoostDataInput import IvyXGBoostDataInput
from IvyXGBoostTrainer import IvyXGBoostTrainer


def write_custom_json(booster, fname, feature_names, class_labels=[]):
   buff = "[\n"
   trees = booster.get_dump()
   ntrees = len(trees)
   for itree,tree in enumerate(trees):
      prev_depth = 0
      depth = 0
      for line in tree.splitlines():
         depth = line.count("\t")
         nascending = prev_depth - depth
         (depth == prev_depth-1)
         # print ascending, depth, prev_depth
         prev_depth = depth
         parts = line.strip().split()
         padding = "   "*depth
         for iasc in range(nascending):
            buff += "{padding}]}},\n".format(padding="   "*(depth-iasc+1))
         if len(parts) == 1:  # leaf
            nodeid = int(parts[0].split(":")[0])
            leaf = float(parts[0].split("=")[-1])
            # print "leaf: ",depth,nodeid,val
            buff += """{padding}{{ "nodeid": {nodeid}, "leaf": {leaf} }},\n""".format(
                  padding=padding,
                  nodeid=nodeid,
                  leaf=leaf,
                  )
         else:
            nodeid = int(parts[0].split(":")[0])
            split, split_condition = parts[0].split(":")[1].replace("[f","").replace("]","").split("<")
            split = feature_names[int(split)]
            split_condition = float(split_condition)
            yes, no, missing = map(lambda x:int(x.split("=")[-1]), parts[1].split(","))
            # print "branch: ",depth,nodeid,split,split_condition,yes,no
            buff += """{padding}{{ "nodeid": {nodeid}, "depth": {depth}, "split": "{split}", "split_condition": {split_condition}, "yes": {yes}, "no": {no}, "missing": {missing}, "children": [\n""".format(
                  padding=padding,
                  nodeid=nodeid,
                  depth=depth,
                  split=split,
                  split_condition=split_condition,
                  yes=yes,
                  no=no,
                  missing=missing,
                  )
      for i in range(depth):
         padding = "   "*(max(depth-1,0))
         if i == 0:
            buff += "{padding}]}}".format(padding=padding)
         else:
            buff += "\n{padding}]}}".format(padding=padding)
         depth -= 1
      if itree != len(trees)-1:
         buff += ",\n"
   buff += "\n]"
   # print buff
   to_dump = {
         "trees": list(ast.literal_eval(buff)),
         "feature_names": feature_names,
         "class_labels": map(int,np.array(class_labels).tolist()), # numpy array not json serializable
         }
   with open(fname, "w") as fout:
      json.dump(to_dump,fout,indent=2)


random.seed(345612)

xx1 = []
xx2 = []
xx3 = []
xx4 = []

n1=1500
n2=3000
n3=15000
n4=7500
for ev in range(0,max(n1,max(n2,max(n3,n4)))):
   do_xx1 = (ev < n1)
   do_xx2 = (ev < n2)
   do_xx3 = (ev < n3)
   do_xx4 = (ev < n4)
   if do_xx1:
      xx1.append([random.betavariate(0.5,0.5)])
   if do_xx2:
      xx2.append([random.betavariate(2.0,5.0)])
   if do_xx3:
      xx3.append([random.betavariate(2.0,2.0)])
   if do_xx4:
      xx4.append([random.betavariate(1.0,3.0)])

xx1 = np.array(xx1, dtype=np.float32)
xx2 = np.array(xx2, dtype=np.float32)
xx3 = np.array(xx3, dtype=np.float32)
xx4 = np.array(xx4, dtype=np.float32)

xgbdata = IvyXGBoostDataInput(["xx"])
xgbdata.add_data(xx1,1.,1,0.5,control_fraction=1./3.)
xgbdata.add_data(xx2,1.,2,0.5,control_fraction=1./3.)
xgbdata.add_data(xx3,1.,0,0.5,control_fraction=1./3.)
xgbdata.add_data(xx4,1.,3,0.5,control_fraction=1./3.)

xgbparams = IvyXGBoostParameters()
xgbparams.setParameters(
   num_round=500, eta=0.3
)

xgbtrainer = IvyXGBoostTrainer()
xgbtrainer.train(xgbdata,xgbparams,early_stopping_rounds=10,scale_weights=True, save_predictions=True)
xgbtrainer.save_model("test_model.bin")
xgbtrainer.save_model("test_model.json")

pred_test = np.array(xgbtrainer.prediction_test, dtype=np.float32)
pred_test = np.column_stack((xgbdata.data_test[0],xgbdata.data_test[2],pred_test))
print("Test sample size: {}".format(pred_test.shape[0]))
print(pred_test)

pred_control = np.array(xgbtrainer.prediction_control, dtype=np.float32)
pred_control = np.column_stack((xgbdata.data_control[0],xgbdata.data_control[2],pred_control))
print("Control sample size: {}".format(pred_control.shape[0]))
print(pred_control)
