from load_dataset import *
from setting import params
car, non_car= load_dataset()
# car=car[:3]
# non_car=non_car[:3]

os.chdir("/Users/datle/Desktop/license_plate_detection/dataset/non-vehicles")
car_feature=extract_feature(car, params['color_space'])
os.chdir("/Users/datle/Desktop/nhan_dien_xe")
non_car_feature= extract_feature(non_car, params['color_space'])
X,y= combine(car_feature, non_car_feature)
sc, X_scaled= normalize(X)
X_train, X_test, y_train, y_test= split(X_scaled, y)
model= train_model(X_train, X_test, y_train, y_test)
save_model('lp_detect.p', model, sc, params=params,y=y)
