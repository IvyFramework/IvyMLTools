import random
import numpy as np
from IvyXGBoostParameters import IvyXGBoostParameters
from IvyXGBoostDataInput import IvyXGBoostDataInput
from IvyXGBoostTrainer import IvyXGBoostTrainer
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plotter
from sklearn.metrics import confusion_matrix
import itertools


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

features = ["xx"]
xgbdata = IvyXGBoostDataInput(features)
xgbdata.add_data(xx1,1.,1,0.5,control_fraction=1./3.)
xgbdata.add_data(xx2,1.,2,0.5,control_fraction=1./3.)
xgbdata.add_data(xx3,1.,0,0.5,control_fraction=1./3.)
xgbdata.add_data(xx4,1.,3,0.5,control_fraction=1./3.)

xgbparams = IvyXGBoostParameters()
xgbparams.setParameters(
   num_round=500, eta=0.05
)

xgbtrainer = IvyXGBoostTrainer()
xgbtrainer.train(xgbdata,xgbparams,early_stopping_rounds=10,scale_weights=True, save_predictions=True)
xgbtrainer.save_model("test_model.bin")
xgbtrainer.save_model("test_model.dump")

dsets_compare = []

pred_train = np.array(xgbtrainer.prediction_train, dtype=np.float32)
pred_train_discrete = np.argmax(pred_train,axis=1)
pred_train = np.column_stack((xgbdata.data_train[0],xgbdata.data_train[2],pred_train))
print("Training sample size: {}".format(pred_train.shape[0]))
print(pred_train)
dsets_compare.append(["Training sample", xgbdata.data_train[2], pred_train_discrete])

pred_test = np.array(xgbtrainer.prediction_test, dtype=np.float32)
pred_test_discrete = np.argmax(pred_test,axis=1)
pred_test = np.column_stack((xgbdata.data_test[0],xgbdata.data_test[2],pred_test))
print("Test sample size: {}".format(pred_test.shape[0]))
print(pred_test)
dsets_compare.append(["Test sample", xgbdata.data_test[2], pred_test_discrete])

pred_control = np.array(xgbtrainer.prediction_control, dtype=np.float32)
pred_control_discrete = np.argmax(pred_control,axis=1)
pred_control = np.column_stack((xgbdata.data_control[0],xgbdata.data_control[2],pred_control))
print("Control sample size: {}".format(pred_control.shape[0]))
print(pred_control)
dsets_compare.append(["Control sample", xgbdata.data_control[2], pred_control_discrete])

# Plot the results
normalize_confMat = True
txtfmt = '.2f' if normalize_confMat else 'd'
fig,panels = plotter.subplots(1,len(dsets_compare))
fig.suptitle("Confusion matrices")
for ipanel in range(len(dsets_compare)):
   panel_title = dsets_compare[ipanel][0]
   true_classes = dsets_compare[ipanel][1]
   predicted_classes = dsets_compare[ipanel][2]
   class_types = np.unique(true_classes)
   panel = panels[ipanel]
   panel.set_title(panel_title)

   confMat = confusion_matrix(true_classes, predicted_classes)
   if normalize_confMat:
      confMat = confMat.astype(np.float32) / confMat.sum(axis=1)[:, np.newaxis]

   panel.imshow(confMat, interpolation='nearest', cmap="Blues")
   #panel.colorbar()
   tick_marks = np.arange(len(class_types))
   panel.set_xticks(tick_marks, map(str,class_types))
   panel.set_yticks(tick_marks, map(str,class_types))

   thresh = 0.6 * confMat.max()
   for i, j in itertools.product(range(confMat.shape[0]), range(confMat.shape[1])):
      val_conf = confMat[i,j]
      panel.text(
         j, i, format(val_conf, txtfmt),
         horizontalalignment="center",
         verticalalignment="center",
         color="white" if val_conf > thresh else "black"
      )
   panel.set(ylabel='True class',xlabel='Predicted class')
for panel in panels:
   panel.label_outer()

fig.set_tight_layout(True)
fig.savefig("mat.png")
