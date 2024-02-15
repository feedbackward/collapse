'''Setup: training routines specialized for different methods.'''

# External modules.
import torch


###############################################################################


def do_training(method, model, optimizer, dl_tr, loss_fn, skip_singles=False):
    '''
    A single training pass over a data loader.
    '''
    model.train()
    if method == "SAM":
        for X, Y in dl_tr:
            if len(X) == 1 and skip_singles:
                continue
            else:
                # First forward-backward pass.
                Y_hat = model(X)
                loss = loss_fn(Y_hat, Y)
                optimizer.zero_grad()
                loss.backward()
                with torch.no_grad():
                    optimizer.first_step()
                # Second forward-backward pass.
                Y_hat = model(X)
                loss = loss_fn(Y_hat, Y)
                optimizer.zero_grad()
                loss.backward()
                with torch.no_grad():
                    optimizer.second_step()
    else:
        for X, Y in dl_tr:
            if len(X) == 1 and skip_singles:
                continue
            else:
                Y_hat = model(X)
                loss = loss_fn(Y_hat, Y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
    return None


###############################################################################
