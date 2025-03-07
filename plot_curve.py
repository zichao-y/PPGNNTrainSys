import os
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt

def extract_tensorboard_data(tensorboard_dir):
    # Create an event accumulator
    ea = event_accumulator.EventAccumulator(tensorboard_dir,
        size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()

    # Extract scalar data
    data = {}
    for tag in ea.Tags()["scalars"]:
        events = ea.Scalars(tag)
        values = [event.value for event in events]
        steps = [event.step for event in events]
        data[tag] = (steps, values)

    return data

# def plot_data(data, out_dir=None):
#     plt.figure(figsize=(10, 5))
    
#     for tag, (steps, values) in data.items():
#         plt.plot(steps, values, label=tag)

#     plt.xlabel("Steps")
#     plt.ylabel("Value")
#     plt.title("Training Curves")
#     plt.legend()
#     plt.savefig(os.path.join(out_dir, 'training_curves.png'))

def plot_data(data, out_dir=None):
    plt.figure(figsize=(10, 5))
    
    ax1 = plt.gca()  # Get current axes
    ax2 = ax1.twinx()  # Create another y-axis that shares the same x-axis

    for tag, (steps, values) in data.items():
        if tag.startswith("Acc/"):  # Check if the tag is for accuracy data
            ax2.plot(steps, values, label=tag)
        else:
            ax1.plot(steps, values, label=tag)

    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Value")
    ax2.set_ylabel("Accuracy")  # Label for the right-hand side y-axis
    ax1.set_title("Training Curves")

    # Handling legends for both axes
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper left')

    plt.savefig(os.path.join(out_dir, 'training_curves.png'))

def plot_curve(dir):
    tensorboard_dir = dir
    data = extract_tensorboard_data(tensorboard_dir)
    plot_data(data, out_dir=dir)
