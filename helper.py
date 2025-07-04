import matplotlib.pyplot as plt

# Initialize once
plt.ion()
fig, ax = plt.subplots()

def plot(scores, mean_scores,save_plot):
    ax.clear()
    ax.set_title('Training...')
    ax.set_xlabel('Number of Games')
    ax.set_ylabel('Score')
    ax.plot(scores, label='Score')
    ax.plot(mean_scores, label='Mean')

    if scores and mean_scores:
        all_values = scores + mean_scores
        y_min = min(all_values)
        y_max = max(all_values)
        padding = (y_max - y_min) * 0.1  # 10% padding
        ax.set_ylim(y_min - padding, y_max + padding)

    ax.text(len(scores)-1, scores[-1], str(scores[-1]))
    ax.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    ax.legend()
    

    if save_plot!=0:
        filename = f"normal_training_plot_episode_{save_plot}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as {filename}")

    plt.pause(0.1)


