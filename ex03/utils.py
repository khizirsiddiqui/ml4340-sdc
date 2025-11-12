import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os


def get_state(state): 
    """ Helper function to transform state """ 
    state = np.ascontiguousarray(state, dtype=np.float32) 
    return np.expand_dims(state, axis=0)

def visualize_training(episode_rewards, training_losses, ourdir =""):
    """ Visualize training by creating reward + loss plots
    Parameters
    -------
    episode_rewards: list
        list of cumulative rewards per training episode
    training_losses: list
        list of training losses
    model_identifier: string
        identifier of the agent
    """
    plt.plot(np.array(episode_rewards))
    plt.savefig(os.path.join(ourdir, "episode_rewards.png"))
    plt.close()
    plt.plot(np.array(training_losses))
    plt.savefig(os.path.join(ourdir, "training_losses.png"))
    plt.close()

def save_training_data(episode_rewards, training_losses, outdir=""):
    """ Save training data to npy files
    Parameters
    -------
    episode_rewards: list
        list of cumulative rewards per training episode
    training_losses: list
        list of training losses
    outdir: string
        output directory
    """
    np.save(os.path.join(outdir, "episode_rewards.npy"), np.array(episode_rewards))
    np.save(os.path.join(outdir, "training_losses.npy"), np.array(training_losses))

def load_actions ( action_filename ):

    actions = []

    with open ( action_filename ) as f:

        lines = f.readlines()

        for line in lines:
            action = []
            for tok in line.split():
                action.append ( float ( tok ))
            actions.append (action)

    return actions
