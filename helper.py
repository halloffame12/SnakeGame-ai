import matplotlib.pyplot as plt
from IPython import display
import os

plt.ion()

def plot(scores, mean_scores):
    # Create the plot directory if it doesn't exist
    plot_dir = './plots'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores, label='Scores', alpha=0.7)
    plt.plot(mean_scores, label='Mean Scores', linewidth=2)
    plt.ylim(ymin=0)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add text annotations
    if len(scores) > 0:
        plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    if len(mean_scores) > 0:
        plt.text(len(mean_scores)-1, mean_scores[-1], f'{mean_scores[-1]:.2f}')
    
    plt.show(block=False)
    plt.pause(.1)
    
    # Save the plot image
    plt.savefig(os.path.join(plot_dir, 'training_plot.png'))