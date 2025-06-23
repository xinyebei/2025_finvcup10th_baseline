'''
description: 
param : 
return: 
Author: xinyebei@xinye.com
Date: 2025-01-14 18:01:28
LastEditors: xinyebei@xinye.com
'''
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics


def parse_metric_for_print(metric_dict):
    if metric_dict is None:
        return "\n"
    str = "\n"
    str += "================================ Each dataset best metric ================================ \n"
    for key, value in metric_dict.items():
        if key != 'avg':
            str= str+ f"| {key}: "
            for k,v in value.items():
                str = str + f" {k}={v} "
            str= str+ "| \n"
        else:
            str += "============================================================================================= \n"
            str += "================================== Average best metric ====================================== \n"
            avg_dict = value
            for avg_key, avg_value in avg_dict.items():
                if avg_key == 'dataset_dict':
                    for key,value in avg_value.items():
                        str = str + f"| {key}: {value} | \n"
                else:
                    str = str + f"| avg {avg_key}: {avg_value} | \n"
    str += "============================================================================================="
    return str


def get_test_metrics(y_pred, y_true, img_names, threshold_acc=0.5):
    def get_video_metrics(image, pred, label):
        result_dict = {}
        new_label = []
        new_pred = []
        # print(image[0])
        # print(pred.shape)
        # print(label.shape)
        for item in np.transpose(np.stack((image, pred, label)), (1, 0)):

            s = item[0]
            if '\\' in s:
                parts = s.split('\\')
            else:
                parts = s.split('/')
            a = parts[-2]
            b = parts[-1]

            if a not in result_dict:
                result_dict[a] = []

            result_dict[a].append(item)
        image_arr = list(result_dict.values())

        for video in image_arr:
            pred_sum = 0
            label_sum = 0
            leng = 0
            for frame in video:
                pred_sum += float(frame[1])
                label_sum += int(frame[2])
                leng += 1
            new_pred.append(pred_sum / leng)
            new_label.append(int(label_sum / leng))
        fpr, tpr, thresholds = metrics.roc_curve(new_label, new_pred)
        v_auc = metrics.auc(fpr, tpr)
        fnr = 1 - tpr
        v_eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        return v_auc, v_eer

    # WARNING: y_pred here is actually the confidence score, not the predicted label, we do not modify the authors origin code, but a warning is given
    y_pred = y_pred.squeeze()
    # For UCF, where labels for different manipulations are not consistent.
    y_true[y_true >= 1] = 1
    # auc
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
    if np.isnan(fpr[0]):
        auc = 0
        eer = 0
    else:
        fnr = 1 - tpr
        auc = metrics.auc(fpr, tpr)
        # 绘制 ROC 曲线
        # plt.figure(figsize=(10, 6))
        # plt.plot(fpr, tpr, marker='.')
        # plt.plot([0, 1], [0, 1], linestyle='--')  # 随机分类器的 ROC 曲线
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title('ROC Curve')
        # plt.legend()
        # plt.grid(True)
        # plt.savefig('roc_curve.png')
        # eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        eer = 0

    # ap y_pred is actually the confidence score, not the predicted label
    ap = metrics.average_precision_score(y_true, y_pred)
    precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_pred)

    
    # acc
    print("====" * 10)
    print(f"acc thresould: {threshold_acc}")
    print("====" * 10)

    
    recalls = [] 
    precisions = []
    f1s = []
    touch_count = [] # 0: 0.5, 1: 0.7, 2: 0.9

    for thred in [0.5, 0.7, 0.9]:
        touch_count.append((y_pred > thred).sum().item())
        
        
    for label in range(2):
        for thred in [0.1, 0.3, 0.5, 0.7, 0.9,]:

            prediction_class = (y_pred > thred).astype(int)
            correct = (prediction_class == np.clip(y_true, a_min=0, a_max=1)).sum().item()
            acc = correct / len(prediction_class)
            f1s.append(metrics.f1_score(y_true, prediction_class, pos_label=label))
            recalls.append(metrics.recall_score(y_true, prediction_class, pos_label=label))
            precisions.append(metrics.precision_score(y_true, prediction_class, pos_label=label))

        
        
    if type(img_names[0]) is not list and type(img_names[0]) is not str:
        # calculate video-level auc for the frame-level methods.
        v_auc, _ = get_video_metrics(img_names, y_pred, y_true)
    else:
        # video-level methods
        v_auc=auc

    return {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap, 'pred': y_pred.tolist(), \
        'video_auc': v_auc, 'label': y_true.tolist(), "cls": prediction_class.tolist(), "img_names": img_names, \

        'real-recall@0.1':recalls[0], 'real-recall@0.3':recalls[1], 'real-recall@0.5':recalls[2], 'real-recall@0.7':recalls[3], 'real-recall@0.9':recalls[4], \
        'real-precision@0.1':precisions[0], 'real-precision@0.3':precisions[1], 'real-precision@0.5':precisions[2], 'real-precision@0.7':precisions[3], 'real-precision@0.9':precisions[4], \
        'real-f1@0.1':f1s[0],'real-f1@0.3':f1s[1], 'real-f1@0.5':f1s[2], 'real-f1@0.7':f1s[3], 'real-f1@0.9':f1s[4], \

        'fake-recall@0.1':recalls[5],'fake-recall@0.3':recalls[6], 'fake-recall@0.5':recalls[7], 'fake-recall@0.7':recalls[8], 'fake-recall@0.9':recalls[9], \
        'fake-precision@0.1':precisions[5],'fake-precision@0.3':precisions[6], 'fake-precision@0.5':precisions[7], 'fake-precision@0.7':precisions[8], 'fake-precision@0.9':precisions[9], \
        'fake-f1@0.1':f1s[5],'fake-f1@0.3':f1s[6], 'fake-f1@0.5':f1s[7], 'fake-f1@0.7':f1s[8], 'fake-f1@0.9':f1s[9], \

        "touch_count@0.5": touch_count[0], "touch_count@0.7": touch_count[1], "touch_count@0.9": touch_count[2], 
        }
    # return {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap, 'video_auc': v_auc}
