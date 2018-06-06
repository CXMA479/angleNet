from sklearn import metrics
import numpy as np
import os
import matplotlib.pyplot as plt


txt_dir = '../data/metric/iou-prob'
outdir = txt_dir
iou_th=.5

model_type_list=['AA_model','AG_model']
file_list = os.listdir(txt_dir)
for model_type in model_type_list:
    label=[]
    score = []
    for txt_file in file_list:
        if model_type not in txt_file:
            continue
        mat = np.loadtxt( os.path.join(txt_dir, txt_file) )
        label +=  list( 1*(mat[:,0]>iou_th) )
        score += list( mat[:,1] )
        fpr,tpr, thresholds = metrics.roc_curve(label, score, pos_label=1)
        precise, recall,_ = metrics.precision_recall_curve(label, score)
#   plt.figure()
    plt.subplot(121)
    color,ls,label=['red','-.','AG'] if 'AG' in model_type else ['black','-','AA']
    plt.plot(fpr, tpr,color,label=label)
    #plt.axes(aspect='equal')
    plt.title(model_type+'ROC')
    plt.subplot(122)
    plt.plot(recall, precise,color,label=label)
    plt.axis('equal')
    plt.title(model_type+'PR')
    savename_roc = os.path.join(outdir,model_type+'-ROC.txt')
    np.savetxt(savename_roc , np.array([fpr,tpr]).T )

    savename_pr = os.path.join(outdir,model_type+'-PR.txt')
    np.savetxt(savename_pr , np.array([recall, precise]).T )
plt.legend()
plt.show()

