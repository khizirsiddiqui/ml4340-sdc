# uses demonstrations.py to load_demonstrations
# uses ClassificationNetwork from network.py to create a neural NN
from network import ClassificationNetwork, index_class 
from demonstrations import load_demonstrations

def load_data():
    obs, act = load_demonstrations("./data")
    return obs, act

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

if __name__ == "__main__":
    observations, actions = load_data()
    compute_data_stats(observations, actions)


