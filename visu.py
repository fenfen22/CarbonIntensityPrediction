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
    plt.xticks(fontsize=20)  
    plt.yticks(fontsize=20)
    plt.legend(loc='upper right',  fontsize=20, shadow=True, fancybox=True)
    plt.tight_layout()

    plt.savefig(name, bbox_inches='tight')
    

def plot_loss_valiloss_training(loss, vali_loss,folder_path):
    plt.figure(figsize=(10, 7))
    plt.plot(loss, marker='o', linestyle='dashdot',color='green', label='training loss')
    plt.plot(vali_loss, marker='o', linestyle='dashed', color='red', label='validation loss')

    plt.xlabel('Epoch',fontsize=20)
    plt.ylabel('Value',fontsize=20)
    plt.xticks(fontsize=20) 
    plt.yticks(fontsize=20) 
    plt.tight_layout()  
  
    plt.legend(loc='upper right',  fontsize=20, shadow=True, fancybox=True)
    plt.savefig(folder_path +"loss.png")
