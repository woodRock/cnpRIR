import random 
import math
import torch 
import itertools
import numpy as np
import matplotlib.pyplot as plt 


def get_context_target_tensor_3D(data, repeat=1, sampling_rate=0.5, min_size=4, cuda=True, deterministic=False): 
    context = []
    context_mask = []
    target_x = []
    target_y = []
    target_mask = []

    for _ in range(repeat):
        for x, y in data:
            assert len(x) == len(y)
            if len(x) < min_size:
                continue

            assert sampling_rate <= 1, "sampling_rate must be in range [0,1]."
            if sampling_rate < 1:
                sample_size = math.ceil(len(x) * sampling_rate) 
                if not deterministic:     
                    idx = random.sample(range(len(x)), sample_size)
                else:
                    idx = list(range(0, sample_size))
                context_x = x[idx]
                context_y = y[idx]
            else:
                context_x = x
                context_y = y

            # TODO: Process data without tuples, and just real part of RIR (y).
            xx = np.array(
                list(
                    list(
                        itertools.chain.from_iterable(x)) 
                    for x in list(zip([list(a) for a in context_x], [[a] for a in context_y]))))
                                                                                    
            context.append(torch.tensor(xx, dtype=torch.float))
            context_mask.append(torch.ones(len(context_x), dtype=torch.long)) 
            target_x.append(torch.tensor(np.array([list(a) for a in x]), dtype=torch.float))
            target_y.append(torch.tensor(np.array([[a for a in y]]).T, dtype=torch.float))
            target_mask.append(torch.ones(len(x), dtype=torch.long))

    context_t = torch.nn.utils.rnn.pad_sequence(context, batch_first=True, padding_value=0) 
    context_mask_t = torch.nn.utils.rnn.pad_sequence(context_mask, batch_first=True, padding_value=0)
    target_x_t = torch.nn.utils.rnn.pad_sequence(target_x, batch_first=True, padding_value=0) 
    target_y_t = torch.nn.utils.rnn.pad_sequence(target_y, batch_first=True, padding_value=0) 
    target_mask_t = torch.nn.utils.rnn.pad_sequence(target_mask, batch_first=True, padding_value=0)

    if cuda:
      context_t = context_t.cuda()
      context_mask_t = context_mask_t.cuda()
      target_x_t = target_x_t.cuda()
      target_y_t = target_y_t.cuda()
      target_mask_t = target_mask_t.cuda()

    return context_t, context_mask_t, target_x_t, target_y_t, target_mask_t


def mse(actual, pred): 
    assert len(actual) == len(pred), "Need the same number of predictions and target labels."
    actual, pred = np.array(actual), np.array(pred)
    return np.square(np.subtract(actual,pred)).mean() 

def n_room_acc(dataset, cnp, n=5, plot=False):
    (context_t, context_mask_t, target_x_t, target_y_t, target_mask_t) = get_context_target_tensor_3D(dataset, sampling_rate=1)
    mu, sigma = cnp(context_t, context_mask_t, target_x_t)

    if plot:
        fig, ax = plt.subplots(1,5, sharex=True, sharey=True, figsize=(15,3))
    i = 0 # Plot index

    mses = []

    for room in random.sample(list(range(len(mu))), n):
        y_pred = mu[room].detach().cpu().numpy().flatten()
        y = target_y_t[room].detach().cpu().numpy().flatten()
        error = mse(y, y_pred)
        mses.append(error)
        # print(f"Mean Squared Error for Room {room}: {error}")
        if plot:
            ax[i].scatter(y, y_pred)
            x = np.linspace(-0.5,0.5,len(y))
            ax[i].plot(x, x, linestyle="--", color="r", label="Perfect")
            ax[i].set_title(f"Room {room}")
        i += 1

    print(f"MSE mean {np.mean(mses)}, variance {np.std(mses)}")

    if plot: 
        plt.setp(ax[1], xlabel='Predicted - Impulse Response Accuracy')
        plt.setp(ax[0], ylabel='Expected')
        plt.xlabel("Predicted")
        plt.legend()
        plt.savefig("room-mse.png")
        plt.show()


def room_confidence_acc(dataset, cnp, room=0):
    fig, ax = plt.subplots(1,5, sharex=True, sharey=True, figsize=(15,3))
    i = 0

    for r in range(1,6):
        sampling_rate = 0.2 * r
        (context_t, context_mask_t, target_x_t, target_y_t, target_mask_t) = get_context_target_tensor_3D(dataset, sampling_rate=sampling_rate, deterministic=True)
        mu, sigma = cnp(context_t, context_mask_t, target_x_t)

        n = int(len(context_t[room])*sampling_rate)
        sigma_ = sigma[room].detach().cpu().numpy().flatten()[0:n]
        y_pred = mu[room].detach().cpu().numpy().flatten()[0:n]
        y = target_y_t[room].detach().cpu().numpy().flatten()[0:n]

        confidence = round(sampling_rate*100, 2)
        error = mse(y, y_pred)
        print(f"{confidence}% confidence MSE: {error}")

        ax[i].scatter(y, y_pred)
        x = np.linspace(-0.5,0.5,len(y))
        ax[i].plot(x, x, linestyle="--", color="r", label="Perfect")
        ax[i].set_title(f"{confidence}% confidence")

        i += 1

    plt.setp(ax[2], xlabel='Predicted - Impulse Response Accuracy')
    plt.setp(ax[0], ylabel='Expected')
    plt.legend()
    plt.savefig("confidence-mse.png")
    plt.show()