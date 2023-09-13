
def add_dataset_args(parser):
    parser.add_argument("--envs", type=int, required=False,
                        default=100000, help="Envs")
    parser.add_argument("--envs_eval", type=int, required=False,
                        default=100, help="Eval Envs")
    parser.add_argument("--hists", type=int, required=False,
                        default=1, help="Histories")
    parser.add_argument("--samples", type=int,
                        required=False, default=1, help="Samples")
    parser.add_argument("--H", type=int, required=False,
                        default=100, help="Context horizon")
    parser.add_argument("--dim", type=int, required=False,
                        default=10, help="Dimension")
    parser.add_argument("--var", type=float, required=False,
                        default=0.0, help="Bandit arm variance")
    parser.add_argument("--cov", type=float, required=False,
                        default=0.0, help="Coverage of optimal arm")
    parser.add_argument("--env", type=str, required=True, help="Environment")
    parser.add_argument("--env_id_start", type=int, required=False,
                        default=-1, help="Start index of envs to sample")
    parser.add_argument("--env_id_end", type=int, required=False,
                        default=-1, help="End index of envs to sample")


def add_model_args(parser):
    parser.add_argument("--embd", type=int, required=False,
                        default=32, help="Embedding size")
    parser.add_argument("--head", type=int, required=False,
                        default=1, help="Number of heads")
    parser.add_argument("--layer", type=int, required=False,
                        default=3, help="Number of layers")
    parser.add_argument("--lr", type=float, required=False,
                        default=1e-3, help="Learning Rate")
    parser.add_argument("--dropout", type=float,
                        required=False, default=0, help="Dropout")
    parser.add_argument('--shuffle', default=False, action='store_true')


def add_train_args(parser):
    parser.add_argument("--num_epochs", type=int, required=False,
                        default=1000, help="Number of epochs")


def add_eval_args(parser):
    parser.add_argument("--epoch", type=int, required=False,
                        default=-1, help="Epoch to evaluate")
    parser.add_argument("--test_cov", type=float,
                        required=False, default=-1.0,
                        help="Test coverage (for bandit)")
    parser.add_argument("--hor", type=int, required=False,
                        default=-1, help="Episode horizon (for mdp)")
    parser.add_argument("--n_eval", type=int, required=False,
                        default=100, help="Number of eval trajectories")
    parser.add_argument("--save_video", default=False, action='store_true')
