
fresdf2 = pd.read_pickle(f'{adir}/fp_cent_mean2.pkl')

fresdf3 = pd.read_pickle(f'{adir}/fp_cent_mean3.pkl')


testapdf[testapdf.Active==1].iloc[:,2:].mean()

testapdf[testapdf.Inactive==1].iloc[:,2:].mean()

aa = pd.read_pickle(f'{adir}/df_result.pkl')

fresdf = pd.read_pickle(f'{adir}/fp_cent_mean.pkl')
final_df = pd.read_pickle(f'{adir}/final_df.pkl')



final_df = final_df.sort_values(['AID','CID'])

fpdf = pd.read_pickle(f'{adir}/aid_fp_stats.pkl')

apdf = pd.read_pickle(f'{adir}/aid_ap_stats.pkl')
fpdf = fpdf.sort_values(['AID','CID'])
apdf = apdf.sort_values(['AID','CID'])

apdf = apdf.reset_index(drop=True)

fpdf = fpdf.reset_index(drop=True)


fresdf = pd.concat([fresdf, fresdf2])
fresdf.to_pickle(f'{adir}/fp_cent_mean.pkl')


testdf = pd.read_pickle(f'{adir}/test_biodf.pkl')

resdf= pd.read_pickle(f'{adir}/ap_cent_mean.pkl')

resdf2= pd.read_pickle(f'{adir}/ap_cent_mean2.pkl')

resdf=  pd.concat([resdf, resdf2])
resdf = resdf.reset_index(drop=True)


resdf = resdf.sort_values(['AID','CID'])


fresdf= fresdf.reset_index(drop=True)

## 여기서는 Active / Inactive만

final_df = pd.read_pickle(f'{adir}/alldf_pivot.pkl')
final_df = final_df.sort_values(['AID','CID'])
final_df = final_df.reset_index(drop=True)


fresdf = pd.read_pickle(f'{adir}/fp_cent_mean.pkl')
# fresdf = fresdf.sort_values(['AID','CID'])
fresdf = fresdf.reset_index(drop=True)



aresdf = pd.read_pickle(f'{adir}/ap_cent_mean.pkl')
aresdf = aresdf.sort_values(['AID','CID'])
aresdf = aresdf.reset_index(drop=True)

asimdf = pd.read_pickle(f'{adir}/df_result_ap.pkl')
asimdf = asimdf.sort_values(['AID','CID'])
asimdf = asimdf.reset_index(drop=True)


fsimdf = pd.read_pickle(f'{adir}/df_result_fp.pkl')
fsimdf = fsimdf.sort_values(['AID','CID'])
fsimdf = fsimdf.reset_index(drop=True)

final_df = final_df[final_df.AID.isin(fresdf.AID)&final_df.CID.isin(fresdf.CID)].reset_index(drop=True)
aresdf = aresdf[aresdf.AID.isin(final_df.AID)&aresdf.CID.isin(final_df.CID)].reset_index(drop=True)
fresdf = fresdf[fresdf.AID.isin(final_df.AID)&fresdf.CID.isin(final_df.CID)].reset_index(drop=True)
asimdf = asimdf[asimdf.AID.isin(final_df.AID)&asimdf.CID.isin(final_df.CID)].reset_index(drop=True)
fsimdf = fsimdf[fsimdf.AID.isin(final_df.AID)&fsimdf.CID.isin(final_df.CID)].reset_index(drop=True)

fresdf.columns = ['AID','CID'] + [f'fp_{i}' for i in fresdf.columns[2:]]
aresdf.columns = ['AID','CID'] + [f'ap_{i}' for i in aresdf.columns[2:]]

sub_fresdf = fresdf.iloc[:,[0,1,4,5,8,9]]#.dropna()
sub_aresdf = aresdf.iloc[:,[4,5,8,9]]#.dropna()

asimdf.columns = ['AID','CID'] + [f'fp_{i}' for i in asimdf.columns[2:]]
fsimdf.columns = ['AID','CID'] + [f'ap_{i}' for i in fsimdf.columns[2:]]



adf = pd.concat([sub_fresdf, sub_aresdf, asimdf.iloc[:,2:], fsimdf.iloc[:,2:], final_df.iloc[:,2:]], axis=1)

adf = adf.dropna()

adf.to_pickle(f'{adir}/all_simdf.pkl')

adf = pd.read_pickle(f'{adir}/all_simdf.pkl')

X_train = adf[~adf.CID.isin(testdf.CID)].iloc[:,list(range(2,18))].sample(frac=1, random_state=7)
y_train = adf[~adf.CID.isin(testdf.CID)]['Active']

X_test = adf[adf.CID.isin(testdf.CID)].iloc[:,list(range(2,18))].sample(frac=1, random_state=7)
y_test = adf[adf.CID.isin(testdf.CID)]['Active']




####


sfresdf = fresdf[fresdf.index.isin(fresdf.iloc[:,[0,1,4,5,8,9]].dropna().index)]
afresdf = fresdf[fresdf.index.isin(fresdf.iloc[:,0:9].dropna().index)]

sffinal = pd.concat([sfresdf.reset_index(drop=True), final_df[(final_df.AID.isin(sfresdf.AID)&final_df.CID.isin(sfresdf.CID))].reset_index(drop=True).iloc[:,2:]],axis=1)

affinal = pd.concat([afresdf.reset_index(drop=True), final_df[(final_df.AID.isin(afresdf.AID)&final_df.CID.isin(afresdf.CID))].reset_index(drop=True).iloc[:,2:]],axis=1)
# affinal = pd.concat([aresdf.reset_index(drop=True), final_df[(final_df.AID.isin(aresdf.AID)&final_df.CID.isin(aresdf.CID))].reset_index(drop=True).iloc[:,2:]],axis=1)
# sfpdf = apdf[(apdf.AID.isin(sffinal.AID))&(apdf.CID.isin(sffinal.CID))].reset_index(drop=True).iloc[:,2:6]
# sffinal = pd.concat([sffinal, sfpdf] ,axis=1)
# affinal = pd.concat([affinal, apdf[(apdf.AID.isin(affinal.AID))&(apdf.CID.isin(affinal.CID))].reset_index(drop=True).iloc[:,2:6]],axis=1)

sss = resdf[resdf.AID.isin(sffinal.AID)&resdf.CID.isin(sffinal.CID)]
aaa = resdf[resdf.AID.isin(affinal.AID)&resdf.CID.isin(affinal.CID)]


ssffinal= pd.concat([sffinal[sffinal.AID.isin(sss.AID)&sffinal.CID.isin(sss.CID)].reset_index(drop=True),sss.reset_index(drop=True)],axis=1)


aaffinal= pd.concat([affinal[affinal.AID.isin(aaa.AID)&affinal.CID.isin(aaa.CID)].reset_index(drop=True),aaa.reset_index(drop=True)],axis=1)
aaffinal = aaffinal.iloc[:,2:]

sffinal = sffinal.iloc[:,[0,1,4,5,8,9,17,18,19,20,12,13,14,15,16]]

affinal = affinal.iloc[:,[0,1,2,3,4,5,6,7,8,9,17,18,19,20,12,13,14,15,16]]


sfresdf = sfresdf.sort_values(['AID','CID'])
afresdf = afresdf.sort_values(['AID','CID'])


resdf = pd.read_pickle(f'{adir}/ap_cent_mean.pkl')
resdf = resdf.reset_index(drop=True)

resdf = resdf.sort_values(['AID','CID'])

sresdf = resdf[resdf.index.isin(resdf.iloc[:,[0,1,4,5,8,9]].dropna().index)]
aresdf = resdf[resdf.index.isin(resdf.iloc[:,0:9].dropna().index)]


sresdf = sresdf.sort_values(['AID','CID'])
aresdf = aresdf.sort_values(['AID','CID'])

# testresdf = resdf[resdf.CID.isin(testdf.CID)]
# testresdf = testresdf.reset_index(drop=True)


# stestresdf = testresdf[testresdf.index.isin(testresdf.iloc[:,[0,1,4,5,8,9]].dropna().index)]
# stestresdf = stestresdf.iloc[:,[0,1,4,5,8,9]]

# afinal = testresdf[testresdf.index.isin(testresdf.iloc[:,0:9].dropna().index)]
# atestresdf = atestresdf.iloc[:,0:9]

# ff =final_df[(final_df.AID.isin(testresdf.AID)&final_df.CID.isin(testresdf.CID))]
# testresdf = testresdf.sort_values(['AID','CID'])

# yyy = pd.concat([testresdf.reset_index(drop=True), ff.reset_index(drop=True).iloc[:,2:]],axis=1)


# yyy[yyy.Active==1].iloc[:,2:].mean()

# yyy[yyy.Inactive==1].iloc[:,2:].mean()


sfinal = pd.concat([sresdf.reset_index(drop=True), final_df[(final_df.AID.isin(sresdf.AID)&final_df.CID.isin(sresdf.CID))].reset_index(drop=True).iloc[:,2:]],axis=1)

afinal = pd.concat([aresdf.reset_index(drop=True), final_df[(final_df.AID.isin(aresdf.AID)&final_df.CID.isin(aresdf.CID))].reset_index(drop=True).iloc[:,2:]],axis=1)
sapdf = apdf[(apdf.AID.isin(sfinal.AID))&(apdf.CID.isin(sfinal.CID))].reset_index(drop=True).iloc[:,2:6]
sfinal = pd.concat([sfinal, sapdf] ,axis=1)
afinal = pd.concat([afinal, apdf[(apdf.AID.isin(afinal.AID))&(apdf.CID.isin(afinal.CID))].reset_index(drop=True).iloc[:,2:6]],axis=1)




sfinal = sfinal.iloc[:,[0,1,4,5,8,9,17,18,19,20,12,13,14,15,16]]

afinal = afinal.iloc[:,[0,1,2,3,4,5,6,7,8,9,17,18,19,20,12,13,14,15,16]]


sfpdf = fpdf[(fpdf.AID.isin(sfinal.AID))&(fpdf.CID.isin(sfinal.CID))].reset_index(drop=True).iloc[:,2:6]

afpdf = fpdf[(fpdf.AID.isin(afinal.AID))&(fpdf.CID.isin(afinal.CID))].reset_index(drop=True).iloc[:,2:6]

sfinal = pd.concat([sfinal, sfpdf], axis=1)

afinal = pd.concat([afinal, afpdf], axis=1)



X_train = aaffinal[~aaffinal.CID.isin(testdf.CID)].iloc[:,list(range(7))+list(range(17,25))]
y_train = aaffinal[~aaffinal.CID.isin(testdf.CID)]['Active']

X_test = aaffinal[aaffinal.CID.isin(testdf.CID)].iloc[:,list(range(7))+list(range(17,25))]
y_test = aaffinal[aaffinal.CID.isin(testdf.CID)]['Active']




import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

X_train = afinal[~afinal.CID.isin(testdf.CID)].iloc[:,list(range(2,14))+list(range(19,23))]
y_train = afinal[~afinal.CID.isin(testdf.CID)]['Active']

X_test = afinal[afinal.CID.isin(testdf.CID)].iloc[:,list(range(2,14))+list(range(19,23))]
y_test = afinal[afinal.CID.isin(testdf.CID)]['Active']





rff = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=5,
    n_jobs=-1,
    random_state=42,
    class_weight="balanced"
)

rff.fit(X_train, y_train)

# 기본 성능 체크
y_pred_proba = rff.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_proba)
print(f"RF ROC-AUC: {auc:.4f}")


import pandas as pd

fi = pd.DataFrame({
    "feature": list(X_train.columns),
    "importance": rff.feature_importances_
}).sort_values("importance", ascending=False)

print(fi)


from sklearn.inspection import permutation_importance
import time
st = time.time()
perm = permutation_importance(
    rff,
    X_test,
    y_test,
    n_repeats=10,
    random_state=42,
    scoring="roc_auc",
    n_jobs=-1
)
print(f'{time.time()-st}')

perm_df = pd.DataFrame({
    "feature": list(X_train.columns),
    "perm_importance_mean": perm.importances_mean,
    "perm_importance_std": perm.importances_std
}).sort_values("perm_importance_mean", ascending=False)

print(perm_df)


import shap

# SHAP용 background (train에서 소량 샘플)
feature_names = X_train.columns.tolist()

background = X_train[np.random.choice(X_train.shape[0], size=min(500, X_train.shape[0]), replace=False)]

background = X_train.sample(frac=1, random_state=42).iloc[:500, :]

explainer = shap.TreeExplainer(rff, background)
shap_values = explainer.shap_values(X_test)

shap_active = shap_values[:, :, 0]
global_importance = np.abs(shap_active).mean(axis=0)


# shap_abs_mean = np.abs(shap_values[1]).mean(axis=1)

shap_df = pd.DataFrame({
    "feature": feature_names if feature_names is not None else np.arange(X_test.shape[1]),
    "shap_importance": global_importance
}).sort_values("shap_importance", ascending=False)

print(shap_df)
shap.summary_plot(
    shap_values[1],
    X_test,
    feature_names=feature_names,
    plot_type="bar"
)

shap.summary_plot(
    shap_values[1],
    X_test,
    feature_names=feature_names
)
