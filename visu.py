from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

def visual(true, preds=None, name='./pic/test.png'):
    """
    Results visualization
    """
    plt.figure(figsize=(10, 7))
    plt.plot(true, marker='o', linestyle='dashdot',color='#3b88bd', label='Ground-Truth')
    if preds is not None:
        plt.plot(preds, marker='o', linestyle='dashed', color='#ff9030', label='Prediction')

    plt.xlabel('time units',fontsize=20)
    plt.ylabel('values',fontsize=20)
    plt.xticks(fontsize=20)  # Adjust the tick label font size
    plt.yticks(fontsize=20)
    plt.legend(loc='upper right',  fontsize=20, shadow=True, fancybox=True)
    plt.tight_layout()

    plt.savefig(name, bbox_inches='tight')
    
def real_pred_training(df, y_train, y_pred_train, len_obv, n_feature):
    names = df.columns
    names_list = names.tolist()
    for i in range(n_feature):   
        plt.figure(figsize=(12, 5))
        plt.plot(y_train[0:len_obv, i], marker='o', linestyle='dashdot',color='green', label='Real Value of training set')
        plt.plot(y_pred_train[0:len_obv, i], marker='o', linestyle='dashed', color='red', label='Predicted Value of training set')
        plt.xlabel('number of observations')
        plt.title("real-pred-train" + str(i))
        plt.ylabel(names_list[i])
        plt.xticks(rotation=45)  
        plt.tight_layout() 
        plt.grid(True) 
        plt.legend()
        plt_name= result_path + "real-pred-train"+str(i)+ ".png"
        plt.savefig( plt_name)
        # fig=plt.figure()
        # plt.savefig( plt_name, format='png')
        # wandb.log({'chart': wandb.Image(fig)})

        # plt.show()
        plt.close()

def plot_loss_valiloss_training(loss, vali_loss,folder_path):
    plt.figure(figsize=(10, 7))
    plt.plot(loss, marker='o', linestyle='dashdot',color='green', label='training loss')
    plt.plot(vali_loss, marker='o', linestyle='dashed', color='red', label='validation loss')

    # plt.title('Line Chart of Value over Time(Training_set vs predicted_set [sr])')
    plt.xlabel('Epoch',fontsize=20)
    plt.ylabel('Value',fontsize=20)
    plt.xticks(fontsize=20) 
    plt.yticks(fontsize=20) 
    plt.tight_layout()  
    # plt.grid(True) 
    plt.legend(loc='upper right',  fontsize=20, shadow=True, fancybox=True)
    plt.savefig(folder_path +"loss.png")
