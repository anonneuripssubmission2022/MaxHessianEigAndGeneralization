from os import makedirs
import torch
import argparse
from architectures import load_architecture
from utilities import *
from data import *

def main(dataset, arch_id, batch_size, lr, opt='sgd', loss='ce', network_seed=0, train_seed=1, rho=0.05, abridged_size=1000, physical_batch_size=DEFAULT_PHYS_BATCH_SIZE, disable_dropout=False, between_iterates=False, save_freq=10):
    path = get_path(dataset, lr, batch_size, arch_id, opt, loss, rho=rho)
    results_dir = os.environ["RESULTS"]
    directory = f"{results_dir}/{path}_{network_seed}"
    if not os.path.isdir(directory):
        print(f"no model found in {directory}")
        return
        
    train_dataset, test_dataset = load_dataset(dataset)
    [loss_fn, acc_fn] = get_loss_and_acc(loss)
    
    torch.manual_seed(network_seed)
    network = load_architecture(arch_id, dataset).cuda()
    network.load_state_dict(torch.load(f"{directory}/snapshot_final"))
    
    optimizer = get_optimizer(opt, network.parameters(), lr, rho=rho)
    sam = opt=='sam'
    
    torch.manual_seed(train_seed)
    network.eval()
    _, test_acc = compute_losses(network, [loss_fn, acc_fn], test_dataset, DEFAULT_BATCH_SIZE)
    
    abridged_train,_ = load_abridged_dataset(dataset,abridged_size)

    if between_iterates:
        train_gd_extra_iter(directory, network, loss_fn, acc_fn, train_dataset, test_dataset, optimizer, batch_size, verbose=True, print_freq=1, physical_batch_size=physical_batch_size, should_save = save_every, sam=sam)
        leading_eig = get_hessian_leading_eigenvalue_between_iterates(network, directory, loss_fn, batch_size, abridged_train, disable_dropout=disable_dropout, gpu_batch_size=physical_batch_size)
    else:
        leading_eig = get_hessian_leading_eigenvalue_at_iterate(network, loss_fn, batch_size, abridged_train, disable_dropout=disable_dropout, gpu_batch_size=physical_batch_size)
        
    print(f"[final test accuracy = {test_acc:.4f}]\t[max hessian eigenvalue = {leading_eig:.4f}]", flush=True)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train using minibatch SGD")
    parser.add_argument("dataset", type=str, choices=DATASETS, help="which dataset to train")
    parser.add_argument("arch_id", type=str, help="which network architectures to train")
    parser.add_argument("batch_size", type=int, help="the batch size")
    parser.add_argument("lr", type=float, help="the learning rate")
    parser.add_argument("--opt", type=str, choices=["sgd", "sam"],
                        help="which optimization algorithm to use", default="sgd")
    parser.add_argument("--loss", type=str, choices=["ce"], help="which loss function to use", default='ce')
    parser.add_argument("--network_seed", type=int, help="the random seed used when initializing the network weights",
                        default=0)
    parser.add_argument("--train_seed", type=int, help="the random seed used at the start of training",
                        default=1)
    parser.add_argument("--rho", type=float, help="SAM hyperparameter", default=0.05)
    parser.add_argument("--physical_batch_size", type=int,
                        help="the maximum number of examples that we try to fit on the GPU at once", default=1000)
    parser.add_argument("--abridged_size", type=int, default=1000,
                        help="when computing the maximum Hessian eigenvalue, use an abridged dataset of this size")
    parser.add_argument("--disable_dropout", type=bool, default=False,
                        help="True when computing the maximum Hessian eigenvalue using network with dropout (False otherwise)")
    parser.add_argument("--between_iterates", type=bool, default=False,
                        help="True when computing the maximum Hessian eigenvalue between iterates (only used for BN networks trained with small learning rates)")
    parser.add_argument("--save_freq", type=int, default=10,
                        help="the frequency at which we save results")
    args = parser.parse_args()

    main(dataset=args.dataset, arch_id=args.arch_id, batch_size=args.batch_size, lr=args.lr, opt=args.opt, loss=args.loss, network_seed=args.network_seed, train_seed=args.train_seed, rho=args.rho, abridged_size=args.abridged_size, physical_batch_size=args.physical_batch_size, disable_dropout=args.disable_dropout, between_iterates=args.between_iterates, save_freq=args.save_freq)
