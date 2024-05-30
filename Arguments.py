import argparse

parser = argparse.ArgumentParser(description="Network parameters")

parser.add_argument(
    "--checkpoints_dir",
    type=str,
    default="Checkpoints/example",
    help="Where the log and model save",
)

# Main parameters
parser.add_argument(
    "--dataset",
    type=str,
    default='S1131',
    help="SKEMPI2, S1131, S4169",
    choices=['SKEMPI2', 'S1131', 'S4169'],
)

parser.add_argument(
    "--splitting",
    type=str,
    default='mutation',
    help="mutation or complex",
    choices=['mutation', 'complex'],
)

parser.add_argument(
    "--device",
    type=str,
    default="cuda:0",
    help="Which gpu/cpu to train on"
)

parser.add_argument(
    "--n_layers_structure",
    type=int,
    default=4,
    help="Number of convolutional layers"
)

parser.add_argument(
    "--emb_dims",
    type=int,
    default=64,
    help="Number features of hidden layers ",
)

parser.add_argument(
    "--radius_structure",
    type=float,
    default=12.0,
    help="Radius to use for the convolution"
)

parser.add_argument(
    "--radius_interface",
    type=float,
    default=6.0,
    help="Radius to use for the convolution"
)

parser.add_argument(
    "--batch_size",
    type=int,
    default=32,
    help="Number of proteins in a batch"
)



parser.add_argument(
    "--seed",
    type=int,
    default=42, help="Random seed")

parser.add_argument(
    "--lr",
    type=float,
    default=0.001,
    help="learning rate",
)