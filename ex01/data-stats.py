# uses demonstrations.py to load_demonstrations
# uses ClassificationNetwork from network.py to create a neural NN
from network import ClassificationNetwork, index_class 
from demonstrations import load_demonstrations

def load_data():
    obs, act = load_demonstrations("./data")
    return obs, act


def get_color_distribution(observations):
    # each observation is a 96x96x3 image
    color_counts = {}
    for obs in observations:
        # convert to tuple for dict key
        color = tuple(obs[0,0,:].tolist())
        color_counts[color] = color_counts.get(color, 0) + 1
    return color_counts


def compute_data_stats(observations, actions):
    net = ClassificationNetwork()
    classes = net.actions_to_classes(actions)
    classes = [c.item() for c in classes]
    total_samples = len(observations)
    print(f"Total samples: {total_samples}")
    class_counts = {}
    for c in classes:
        class_counts[c] = class_counts.get(c, 0) + 1
    for c, count in class_counts.items():
        print(f"  Class {index_class[c]}: {count}")

    print("Action distribution:")
    action_to_cname = {}
    for act in actions:
        act_tuple = tuple(act)
        class_name = index_class[net.actions_to_classes([ act ])[0].item()]
        if act_tuple not in action_to_cname:
            action_to_cname[act_tuple] = class_name
    action_counts = {}
    for act in actions:
        act_tuple = tuple(act)
        action_counts[act_tuple] = action_counts.get(act_tuple, 0) + 1
    for act, count in action_counts.items():
        print(f"  Action {action_to_cname[act]} {act}: {count}")

    print("Color distribution (top 10):")
    color_counts = get_color_distribution(observations)
    sorted_colors = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)
    for color, count in sorted_colors[:10]:
        print(f"  Color {color}: {count}")
    # plot color distribution
    try:
        import matplotlib.pyplot as plt
        colors = [color for color, _ in sorted_colors[:10]]
        counts = [count for _, count in sorted_colors[:10]]
        plt.bar(range(len(colors)), counts, color=[(r/255, g/255, b/255) for r, g, b in colors])
        plt.xticks(range(len(colors)), [str(color) for color in colors], rotation=45)
        plt.ylabel("Count")
        plt.title("Top 10 Colors in Observations")
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("matplotlib not installed, skipping color distribution plot.")


if __name__ == "__main__":
    observations, actions = load_data()
    compute_data_stats(observations, actions)

# Total samples: 62445
#   Class steer_right: 7007
#   Class steer_left: 11339
#   Class no_action: 25940
#   Class steer_left_gas: 1656
#   Class gas: 15274
#   Class steer_right_gas: 1104
#   Class brake: 125
# Action distribution:
#   Action steer_right (1.0, 0.0, 0.0): 6600
#   Action steer_left (-1.0, 0.0, 0.0): 10856
#   Action no_action (0.0, 0.0, 0.0): 25940
#   Action steer_left_gas (-1.0, 0.5, 0.0): 1656
#   Action gas (0.0, 0.5, 0.0): 15274
#   Action steer_right (1.0, 0.0, 0.8): 407
#   Action steer_right_gas (1.0, 0.5, 0.0): 1104
#   Action steer_left (-1.0, 0.0, 0.8): 483
#   Action brake (-1.0, 0.5, 0.8): 52
#   Action brake (0.0, 0.0, 0.8): 62
#   Action brake (1.0, 0.5, 0.8): 8
#   Action brake (0.0, 0.5, 0.8): 3
# Color distribution (top 10):
#   Color (100, 202, 100): 39491
#   Color (100, 228, 100): 12059
#   Color (100, 100, 100): 6101
#   Color (102, 102, 102): 742
#   Color (105, 105, 105): 711
#   Color (100, 203, 100): 229
#   Color (100, 227, 100): 134
#   Color (0, 0, 0): 126
#   Color (100, 204, 100): 106
#   Color (100, 205, 100): 103
