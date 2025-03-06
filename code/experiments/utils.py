import matplotlib.pyplot as plt

# Function to plot ROC curve
def plot_roc_curve(fpr, tpr, roc_auc, strategy_title, save_path):
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')  # Diagonal line for random guess
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{strategy_title} ROC Curve')
    plt.grid(alpha=0.15)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

def zigzag_append(lists):
    return [item for group in zip(*lists) for item in group]

def chunk_list(lst, n):
    return [lst[i:i + n] for i in range(0, len(lst), n)]