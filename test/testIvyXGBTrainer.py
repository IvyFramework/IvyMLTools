import random
import numpy as np
from IvyXGBoostParameters import IvyXGBoostParameters
from IvyXGBoostDataInput import IvyXGBoostDataInput
from IvyXGBoostTrainer import IvyXGBoostTrainer


xx1 = []
xx2 = []
xx3 = []
xx4 = []

n1=1000
n2=2000
n3=10000
n4=5000
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

ww1 = np.ones(n1, dtype=np.float32)
ww2 = np.ones(n2, dtype=np.float32)
ww3 = np.ones(n3, dtype=np.float32)
ww4 = np.ones(n4, dtype=np.float32)

cl1 = np.full(n1, 1, dtype=np.int32)
cl2 = np.full(n2, 2, dtype=np.int32)
cl3 = np.full(n3, 0, dtype=np.int32)
cl4 = np.full(n4, 3, dtype=np.int32)

xgbdata = IvyXGBoostDataInput(["xx"])
xgbdata.add_data(xx1,ww1,cl1,0.5,False)
xgbdata.add_data(xx2,ww2,cl2,0.5,False)
xgbdata.add_data(xx3,ww3,cl3,0.5,False)
xgbdata.add_data(xx4,ww4,cl4,0.5,False)

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
print(pred_test)
