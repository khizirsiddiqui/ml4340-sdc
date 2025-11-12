import os
import gymnasium as gym
import deepq
import argparse
import platform
import time

from sdc_wrapper import SDC_Wrapper

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

def get_experiment_name():
    '''
     reads the git-commit-hash.txt file with contents as
            git-commit-hash: <hash>
            git-commit-log: Ex03/ABC - <log>
     return the ABC part of git-commit-log
    '''
    experiment_name = "unknown"
    try:
        with open("git-commit-hash.txt", "r") as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("git-commit-log:"):
                    experiment_name = line.split(":", 1)[1].strip().split(" - ")[0]
                    experiment_name = experiment_name.split("/")[-1]
                    break
    except FileNotFoundError:
        print("git-commit-hash.txt file not found.")
    return experiment_name


def main ():

    """ 
    Train a Deep Q-Learning agent in headless mode on the cluster
    """ 

    print ("python version:\t{0}".format (platform.python_version()))
    print ("gym version:\t{0}".format(gym.__version__))

    # get args
    parser = argparse.ArgumentParser()

    parser.add_argument ( '--total_timesteps', type=int, default=100000, help = 'The number of env steps to take' )
    parser.add_argument ( '--action_repeats', type=int, default=4, help='Update the model every action_repeatss steps' )
    parser.add_argument ( '--gamma', type=float, default=0.99, help='selection action on every n-th frame and repeat action for intermediate frames' )
    parser.add_argument ( '--action_filename', type=str, default = 'default_actions.txt', help='a list of actions' )
    parser.add_argument ( '--use_doubleqlearning', default=False, action="store_true", help='a flag that indicates the use of double q learning' )
    parser.add_argument ( '--no_display', default=False, action="store_true", help='a flag indicating whether training runs on the cluster' )
    parser.add_argument ( '--agent_name', type=str, default='agent', help='an agent name' )
    parser.add_argument ( '--outdir', type=str, default='', help='a directory for output' )

    args = parser.parse_args()

    if args.outdir == '':
        experiment_name = get_experiment_name()
        args.outdir = f"results/{experiment_name}"
        if not os.path.exists(args.outdir):
            os.makedirs(args.outdir)
    agent_path = os.path.join(args.outdir, args.agent_name + '.pth')
    if os.path.exists(agent_path):
        raise FileExistsError(f"Error: agent file {agent_path} already exists and will be overwritten."
                                " Please use a different agent_name or outdir.")
    print(f"Creating symbolic link to agent file: {agent_path} -> ./latest.pth")
    try:
        os.unlink('latest.pth')
    except FileNotFoundError:
        pass
    os.symlink(agent_path, 'latest.pth')

    # check args
    print ( "\nArgs information")
    print ( "total_timesteps:     {0}".format ( args.total_timesteps ) )
    print ( "action_repeats:      {0}".format ( args.action_repeats ) )
    print ( "gamma:               {0}".format ( args.gamma ) )
    print ( "action_filename:     {0}".format ( args.action_filename ) )
    print ( "use_doubleqlearning: {0}".format ( "doubleq" if args.use_doubleqlearning else "no doubleq" ) )
    print ( "no display:             {0}".format ( "true" if args.no_display else "false" ) )
    print ( "agent_name:          {0}".format ( args.agent_name ) )
    print ( "outdir:              {0}".format ( args.outdir ) )

    # load actions
    actions = load_actions ( args.action_filename ) 
    print ( "actions:\t\t", actions )

    # start training
    print ( "\nStart training..." )
    render_mode = 'rgb_array' if args.no_display else 'human'
    env = SDC_Wrapper(gym.make('CarRacing-v2', render_mode=render_mode), remove_score=True, return_linear_velocity=False)

    deepq.learn (
                    env, total_timesteps = args.total_timesteps,
                    action_repeat = args.action_repeats,
                    gamma = args.gamma,
                    model_identifier = args.agent_name,
                    outdir= args.outdir,
                    new_actions = actions,
                    use_doubleqlearning = args.use_doubleqlearning
                )
    
    # wrap up
    env.close ()

if __name__ == '__main__':
    main()
