#!/usr/bin/env python
# _*_coding:utf-8_*_
import pandas as pd
import joblib

from feature_scripts.CKSAAGP import get_CKSAAGP
from feature_scripts.APAAC import get_APAAC
from feature_scripts.CTriad import get_CTriad
from feature_scripts.CTDD import get_CTDD
from feature_scripts.GAAC import get_GAAC
from feature_scripts.Moran import get_Moran
from feature_scripts.DDE import get_DDE
from feature_scripts.PAAC import get_PAAC
from feature_scripts.QSOrder import get_QSOrder
from feature_scripts.SOCNumber import get_SOCNumber
def get_proba_feature(fastas):
    CKSAAGP_feature = pd.DataFrame(get_CKSAAGP(fastas))
    APAAC_feature = pd.DataFrame(get_APAAC(fastas))
    CTriad_feature = pd.DataFrame(get_CTriad(fastas))
    CTDD_feature = pd.DataFrame(get_CTDD(fastas))
    GAAC_feature = pd.DataFrame(get_GAAC(fastas))
    Moran_feature = pd.DataFrame(get_Moran(fastas))
    DDE_feature = pd.DataFrame(get_DDE(fastas))
    PAAC_feature = pd.DataFrame(get_PAAC(fastas))
    QSOrder_feature = pd.DataFrame(get_QSOrder(fastas))
    SOCNumber_feature = pd.DataFrame(get_SOCNumber(fastas))
    temp_feature = []
    scale1 = joblib.load("./models/APAACscaler.pkl")
    model1 = joblib.load("./models/APAAC_SVM.pkl")
    x1 = scale1.transform(APAAC_feature)
    y_pred_proba1 = model1.predict_proba(x1)
    temp_feature.append(y_pred_proba1[:,1])

    scale2 = joblib.load("./models/CTDDscaler.pkl")
    model2 = joblib.load("./models/CTDD_LGBM.pkl")
    x2 = scale2.transform(CTDD_feature)
    y_pred_proba2 = model2.predict_proba(x2)
    temp_feature.append(y_pred_proba2[:,1])

    scale3 = joblib.load("./models/CTDDscaler.pkl")
    model3 = joblib.load("./models/CTDD_SVM.pkl")
    x3 = scale3.transform(CTDD_feature)
    y_pred_proba3 = model3.predict_proba(x3)
    temp_feature.append(y_pred_proba3[:,1])

    scale4 = joblib.load("./models/PAACscaler.pkl")
    model4 = joblib.load("./models/PAAC_RF.pkl")
    x4 = scale4.transform(PAAC_feature)
    y_pred_proba4 = model4.predict_proba(x4)
    temp_feature.append(y_pred_proba4[:,1])

    scale5 = joblib.load("./models/CTDDscaler.pkl")
    model5 = joblib.load("./models/CTDD_RF.pkl")
    x5 = scale5.transform(CTDD_feature)
    y_pred_proba5 = model5.predict_proba(x5)
    temp_feature.append(y_pred_proba5[:,1])

    scale6 = joblib.load("./models/CKSAAGPscaler.pkl")
    model6 = joblib.load("./models/CKSAAGP_SVM.pkl")
    x6 = scale6.transform(CKSAAGP_feature)
    y_pred_proba6 = model6.predict_proba(x6)
    temp_feature.append(y_pred_proba6[:,1])

    scale7 = joblib.load("./models/CKSAAGPscaler.pkl")
    model7 = joblib.load("./models/CKSAAGP_XGBT.pkl")
    x7 = scale7.transform(CKSAAGP_feature)
    y_pred_proba7 = model7.predict_proba(x7)
    temp_feature.append(y_pred_proba7[:,1])

    scale8 = joblib.load("./models/DDEscaler.pkl")
    model8 = joblib.load("./models/DDE_SVM.pkl")
    x8 = scale8.transform(DDE_feature)
    y_pred_proba8 = model8.predict_proba(x8)
    temp_feature.append(y_pred_proba8[:,1])

    scale9 = joblib.load("./models/APAACscaler.pkl")
    model9 = joblib.load("./models/APAAC_LGBM.pkl")
    x9 = scale9.transform(APAAC_feature)
    y_pred_proba9 = model9.predict_proba(x9)
    temp_feature.append(y_pred_proba9[:,1])

    scale10 = joblib.load("./models/QSOrderscaler.pkl")
    model10 = joblib.load("./models/QSOrder_SVM.pkl")
    x10 = scale10.transform(QSOrder_feature)
    y_pred_proba10 = model10.predict_proba(x10)
    temp_feature.append(y_pred_proba10[:,1])

    scale11 = joblib.load("./models/CTDDscaler.pkl")
    model11 = joblib.load("./models/CTDD_XGBT.pkl")
    x11 = scale11.transform(CTDD_feature)
    y_pred_proba11 = model11.predict_proba(x11)
    temp_feature.append(y_pred_proba11[:,1])

    scale12 = joblib.load("./models/GAACscaler.pkl")
    model12 = joblib.load("./models/GAAC_SVM.pkl")
    x12 = scale12.transform(GAAC_feature)
    y_pred_proba12 = model12.predict_proba(x12)
    temp_feature.append(y_pred_proba12[:,1])

    scale13 = joblib.load("./models/APAACscaler.pkl")
    model13 = joblib.load("./models/APAAC_RF.pkl")
    x13 = scale13.transform(APAAC_feature)
    y_pred_proba13 = model13.predict_proba(x13)
    temp_feature.append(y_pred_proba13[:,1])

    scale14 = joblib.load("./models/Moranscaler.pkl")
    model14 = joblib.load("./models/Moran_XGBT.pkl")
    x14 = scale14.transform(Moran_feature)
    y_pred_proba14 = model14.predict_proba(x14)
    temp_feature.append(y_pred_proba14[:,1])

    scale15 = joblib.load("./models/SOCNscaler.pkl")
    model15 = joblib.load("./models/SOCN_SVM.pkl")
    x15 = scale15.transform(SOCNumber_feature)
    y_pred_proba15 = model15.predict_proba(x15)
    temp_feature.append(y_pred_proba15[:,1])

    scale16 = joblib.load("./models/CTriadscaler.pkl")
    model16 = joblib.load("./models/CTriad_SVM.pkl")
    x16 = scale16.transform(CTriad_feature)
    y_pred_proba16 = model16.predict_proba(x16)
    temp_feature.append(y_pred_proba16[:,1])



    temp_feature = pd.DataFrame(temp_feature)
    proba_feature = pd.DataFrame(temp_feature.values.T)
    #proba_feature.to_csv("final_feature.csv", index=0)
    return proba_feature


