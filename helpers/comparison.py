import numpy as np
import matplotlib.pyplot as plt

def plot_accuracy_comparison(self, accuracy_no_device, accuracy_with_device):
    """
    Plots a comparison of model accuracy with and without device embeddings.
   """
    categories = ['Seen Devices', 'Unseen Devices']
    x = np.arange(len(categories))  # Label locations
    width = 0.35  # Width of the bars

    fig, ax = plt.subplots(figsize=(8, 5))
    rects1 = ax.bar(x - width/2, accuracy_no_device, width, label='Without Device Embeddings')
    rects2 = ax.bar(x + width/2, accuracy_with_device, width, label='With Device Embeddings')

    # Labels, title, and legend
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Model Accuracy Comparison: With vs Without Device Embeddings')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()

    # Show values on bars
    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height}%', 
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # Offset above bar
                        textcoords="offset points",
                        ha='center', va='bottom')
    plt.show()  