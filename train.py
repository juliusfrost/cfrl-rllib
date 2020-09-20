from ray.rllib.train import create_parser, run

import envs

envs.register()

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    run(args, parser)
