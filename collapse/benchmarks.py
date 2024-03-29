'''Main script: Benchmark data tests.'''

# External modules.
from argparse import ArgumentParser
import mlflow
import numpy as np
import os
import torch

# Internal modules.
from setup.data import get_dataloader
from setup.directories import models_path
from setup.eval import run_evals, run_gradnorm_evals
from setup.losses import get_loss
from setup.models import get_model
from setup.optimizers import get_optimizer
from setup.training import do_training
from setup.utils import makedir_safe


###############################################################################


def get_parser():
    parser = ArgumentParser(
        prog="synthetic",
        description="Benchmark data tests.",
        add_help=True
    )
    parser.add_argument("--adaptive",
                        help="Set to 'yes' for adaptive SAM.",
                        type=str)
    parser.add_argument("--base-gpu-id",
                        default=0,
                        help="Specify which GPU should be the base GPU.",
                        type=int)
    parser.add_argument("--bs-tr",
                        help="Batch size for training data loader.",
                        type=int)
    parser.add_argument("--dataset",
                        help="Dataset name.",
                        type=str)
    parser.add_argument("--dimension",
                        help="Dimension of inputs.",
                        type=int)
    parser.add_argument("--epochs",
                        help="Number of epochs in training loop.",
                        type=int)
    parser.add_argument("--eta",
                        help="Weight parameter on theta in SoftAD.",
                        type=float)
    parser.add_argument("--flood-level",
                        help="Flood level parameter for Ishida method.",
                        type=float)
    parser.add_argument("--force-cpu",
                        help="Either yes or no; parse within main().",
                        type=str)
    parser.add_argument("--force-one-gpu",
                        help="Either yes or no; parse within main().",
                        type=str)
    parser.add_argument("--gradnorm",
                        help="Measure gradients norms? Either yes or no.",
                        type=str)
    parser.add_argument("--imbalance-factor",
                        help="(majority size) / (minority size).",
                        type=float)
    parser.add_argument("--loss",
                        help="Name of the base loss function.",
                        type=str)
    parser.add_argument("--method",
                        help="Abstract method name.",
                        type=str)
    parser.add_argument("--model",
                        help="Model name.",
                        type=str)
    parser.add_argument("--momentum",
                        help="Momentum parameter for optimizers.",
                        type=float)
    parser.add_argument("--noise-frac",
                        help="Fraction of labels to randomly flip.",
                        type=float)
    parser.add_argument("--num-classes",
                        help="Number of classes (for classification tasks).",
                        type=int)
    parser.add_argument("--num-minority-classes",
                        help="Number of minority classes.",
                        type=int)
    parser.add_argument("--optimizer",
                        help="Optimizer name.",
                        type=str)
    parser.add_argument("--optimizer-base",
                        help="Base optimizer name (only for SAM).",
                        type=str)
    parser.add_argument("--pre-trained",
                        help="Specify pre-trained weights.",
                        type=str)
    parser.add_argument("--quantile-level",
                        help="Quantile level parameter for CVaR.",
                        type=float)
    parser.add_argument("--radius",
                        help="Radius parameter (SAM or DRO).",
                        type=float)
    parser.add_argument("--random-seed",
                        help="Integer-valued random seed.",
                        type=int)
    parser.add_argument("--saving-freq",
                        help="Frequency at which to save models.",
                        type=int)
    parser.add_argument("--sigma",
                        help="Scaling parameter for SoftAD.",
                        type=float)
    parser.add_argument("--skip-singles",
                        help="Specify if we need to skip single batches.",
                        type=str)
    parser.add_argument("--step-size",
                        help="Step size parameter for optimizers.",
                        type=float)
    parser.add_argument("--theta",
                        help="Shift parameter for SoftAD.",
                        type=float)
    parser.add_argument("--tilt",
                        help="Tilt parameter for Tilted ERM.",
                        type=float)
    parser.add_argument("--tr-frac",
                        help="Fraction of data not used for validation.",
                        type=float)
    parser.add_argument("--weight-decay",
                        help="Weight decay parameter for optimizers.",
                        type=float)
    return parser


def get_args():
    parser = get_parser()
    return parser.parse_args()


def main(args):

    # Organize clerical arguments.
    force_cpu = True if args.force_cpu == "yes" else False
    force_one_gpu = True if args.force_one_gpu == "yes" else False
    base_gpu_id = args.base_gpu_id
    skip_singles = True if args.skip_singles == "yes" else False
    gradnorm = True if args.gradnorm == "yes" else False
    saving_freq = int(args.saving_freq)
    save_models = True if saving_freq > 0 else False
    saving_counter = 0 # only used if save_models is True
    if save_models:
        makedir_safe(models_path)
    
    # Device setup.
    if force_cpu and not force_one_gpu:
        device = torch.device(type="cpu")
    elif force_one_gpu and not force_cpu:
        device = torch.device(type="cuda", index=base_gpu_id)
    else:
        raise ValueError("Please specify either CPU or single GPU setting.")
    
    # Seed the random generator (numpy and torch).
    rg = np.random.default_rng(args.random_seed)
    rg_torch = torch.manual_seed(seed=args.random_seed)
    
    # Get the data (placed on desired device).
    dataset_paras = {
        "rg": rg,
        "dimension": args.dimension,
        "bs_tr": args.bs_tr,
        "tr_frac": args.tr_frac,
        "noise_frac": args.noise_frac,
        "imbalance_factor": args.imbalance_factor,
        "num_minority_classes": args.num_minority_classes
    }
    dl_tr, eval_dl_tr, eval_dl_va, eval_dl_te = get_dataloader(
        dataset_name=args.dataset,
        dataset_paras=dataset_paras,
        device=device
    )
    
    # Initialize the model (placed on desired device).
    model_paras = {
        "rg": rg,
        "dimension": args.dimension,
        "num_classes": args.num_classes,
        "pre_trained": args.pre_trained
    }
    model = get_model(model_name=args.model, model_paras=model_paras)
    print("Model:", model)
    model = model.to(device)

    # Get the loss function ready.
    loss_paras = {"method": args.method,
                  "flood_level": args.flood_level,
                  "quantile_level": args.quantile_level,
                  "radius": args.radius,
                  "tilt": args.tilt,
                  "theta": args.theta,
                  "sigma": args.sigma,
                  "eta": args.eta}
    loss_fn = get_loss(loss_name=args.loss,
                       loss_paras=loss_paras,
                       device=device)
    print("loss_fn:", loss_fn)
    
    # Set up the optimizer.
    if args.method in ["CVaR", "DRO"]:
        extra_paras = True
    else:
        extra_paras = False
    opt_paras = {"momentum": args.momentum,
                 "step_size": args.step_size,
                 "weight_decay": args.weight_decay,
                 "extra_paras": extra_paras,
                 "adaptive": args.adaptive,
                 "radius": args.radius,
                 "optimizer_base": args.optimizer_base}
    optimizer = get_optimizer(opt_name=args.optimizer,
                              opt_paras=opt_paras,
                              model=model,
                              loss_fn=loss_fn)
    print("Optimizer:", optimizer)
    
    # Execute the training loop.
    for epoch in range(-1, args.epochs):

        print("Epoch: {}".format(epoch))
        
        # Do training step, except at initial epoch.
        if epoch >= 0:
            do_training(method=args.method,
                        model=model,
                        optimizer=optimizer,
                        dl_tr=dl_tr,
                        loss_fn=loss_fn,
                        skip_singles=skip_singles)

        # Store the number of training points from each class.
        if epoch == -1:
            for X, Y in eval_dl_tr:
                unique_labels, label_counts = torch.unique(
                    Y, return_counts=True
                )
                unique_labels = unique_labels.numpy(force=True)
                label_counts = label_counts.numpy(force=True)
                label_count_dict = {}
                for i, label in enumerate(unique_labels):
                    label_count_dict[str(label)] = label_counts[i]
            print("Label counts:")
            print(label_count_dict)
            mlflow.log_params(label_count_dict)

        # Save model if desired.
        if save_models:
            to_save = epoch==-1 or (epoch>0 and (epoch+1)%saving_freq == 0)
        else:
            to_save = False
        if to_save:
            fname_model = os.path.join(
                models_path, "{}_{}_{}.pth".format(args.dataset,
                                                   args.method,
                                                   saving_counter)
            )
            torch.save(model.state_dict(), fname_model)
            saving_counter += 1
        
        # Evaluation step.
        model.eval()
        with torch.no_grad():
            metrics = run_evals(
                model=model,
                data_loaders=(eval_dl_tr, eval_dl_va, eval_dl_te),
                loss_name=args.loss,
                loss_paras=loss_paras
            )
        if gradnorm:
            gradnorm_metrics = run_gradnorm_evals(
                model=model,
                optimizer=optimizer,
                data_loaders=(eval_dl_tr, eval_dl_va, eval_dl_te),
                loss_name=args.loss,
                loss_paras=loss_paras
            )
            metrics.update(gradnorm_metrics)
        
        # Log the metrics of interest.
        mlflow.log_metrics(step=epoch+1, metrics=metrics)

    # Finished.
    return None


if __name__ == "__main__":
    args = get_args()
    main(args)


###############################################################################
