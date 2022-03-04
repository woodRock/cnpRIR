"""
CNP 
===

We use a Conditional Neural Processeses (CNPs) to generate room impulse repsonses (RIR) in the frequency domain. 
A CNP consists of an encoder and decoder architecture, with latent layer representations of the sound field for a room.
The model recieves the room dimensions, source and reciever locations as features. 
The model is trained using context, context is a sound field for a given room, with a certain number of RIRs masked. 
We train the model to predict the RIR at the masked locations, conditioned on the RIRs at known locations. 

References:
1. Garnelo, M., Rosenbaum, D., Maddison, C., Ramalho, T., Saxton, D., Shanahan, M., ... 
    & Eslami, S. A. (2018, July). Conditional neural processes. In International 
    Conference on Machine Learning (pp. 1704-1713). PMLR.
2. Lehmann, E. A., & Johansson, A. M. (2008). Prediction of energy decay in room 
    impulse responses simulated with an image-source model. The Journal of the 
    Acoustical Society of America, 124(1), 269-277.
3. Ian Goodfellow (2018). Jeff Heaton, Yoshua Bengio, and Aaron Courville: 
    Deep learning. Genetic Programming and Evolvable Machines, 19(1), 305-307.
4. Glorot, X., & Bengio, Y. (2010, March). Understanding the difficulty of training 
    deep feedforward neural networks. In Proceedings of the thirteenth international 
    conference on artificial intelligence and statistics (pp. 249-256). JMLR Workshop 
    and Conference Proceedings.
"""

import torch
import torch.nn as nn
from torch.distributions import Normal

class CNP_encoder(nn.Module):
    def __init__(self, layers=(4, 128, 128, 128 ,128), input_dim=3, dropout=None):
        """
        Initialize the encoder.

        Constructs a sequential model of linear layers, with ReLU activations.
        If specied, a dropout layer is added after each activation layer.
        The last two layers are fully connected linear layers to construct the latent layer.

        Parameters:
            layers (tuple): A type of integers specifying the number of nodes in each layer of the encoder.
            input_dim (int): The dimension of the input.
            dropout (float): The dropout rate. Defaults to None (i.e. P = 0).

        """
        super(CNP_encoder, self).__init__()

        assert layers[0] == input_dim + 1, f"First layer must be of shape input_dim + 1 ({input_dim + 1}), instead found shape{layers[0]}."
        self.input_dim = input_dim
        
        modules = []
        for i in range(0, len(layers)-2):
            modules.append(nn.Linear(layers[i], layers[i+1]))
            modules.append(nn.ReLU())
            if dropout is not None:
                modules.append(nn.Dropout(p=dropout))
        
        modules.append(nn.Linear(layers[-2], layers[-1]))
        self.layers = nn.Sequential(*modules)

        
    def forward(self, context, context_mask):     
        """
        A forward pass through the encoder.
        This takes a room as context, and encodes the sound field 
        in a latent layer representation. 
        """
        h = self.layers(context)
        n = torch.stack( [torch.sum(context_mask, 1)] * h.shape[-1], dim=1)
        batch_size, seq_size = context_mask.shape
        x_mask= torch.stack([context_mask]* h.shape[-1] ,dim=2)
        h = torch.sum(h,1) / n.float()
        return h
    
    
class CNP_decoder(nn.Module):
    def __init__(self, layers=(128, 128, 128, 128), input_dim=3, dropout=None):
        """
        Initialize the decoder.

        Constructs a sequential model of linear layers, with ReLU activations.
        If specied, a dropout layer is added after each activation layer.
        The first two layers are two fully connected linear layers to decode the latent layer. 
        The last two layers are two fully connected layers to predict mean and std RIRs. 

        Parameters:
            layers: A tuple of integers specifying the number of nodes in each layer.
            input_dim: The dimension of the input to the decoder.
            dropout: The dropout probability. Defaults to None (i.e. P = 0). 
        """
        super(CNP_decoder, self).__init__()
        
        modules = [nn.Linear(layers[0]+input_dim, layers[1])] 
        modules.append(nn.ReLU())
        
        for i in range(1, len(layers)-1):
            modules.append(nn.Linear(layers[i], layers[i+1]))
            modules.append(nn.ReLU())
            if dropout is not None:
                modules.append(nn.Dropout(p=dropout))
        
        modules.append(nn.Linear(layers[-1], 2)) 
        self.layers = nn.Sequential(*modules)
      
    
    def forward(self, latent_target):   
        """
        A forward pass through the decoder. 
        This takes a latent layer and returns the mean and std
        RIR for the target locations. 
        """
        h = self.layers(latent_target)
        mu = h[:,:,0]
        log_sigma = h[:,:,1]
        sigma = 0.1 + 0.9 * nn.functional.softplus(log_sigma)
        return mu, sigma

    
class CNP(nn.Module):
    def __init__(self, encoder_layers=(7, 128, 128, 128 ,128),
                       decoder_layers=(128, 128, 128, 128),
                       input_dim=3, dropout=None):
        """
        Args:
            encoder_layers (tuple): The number of nodes for each layer of the encoder network. 
            decoder_layers (tuple): The number of nodes for each layer of the decoder network.
            input_dim (int): The number of dimensions of the input.
            dropout (float): The dropout probability. Defaults to None. 
        """
        super(CNP, self).__init__()
        self.encoder = CNP_encoder(layers=encoder_layers, input_dim=input_dim, dropout=dropout)
        self.decoder = CNP_decoder(layers=decoder_layers, input_dim=input_dim, dropout=dropout)

        
    def init_weights(self):
        """
        The goal of Xavier Initialization is to initialize the weights such that 
        the variance of the activations are the same across every layer. 
        This constant variance helps prevent the gradient from exploding or vanishing (Xavier 2010)
        """
        def init_weights_(m):
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight)
        self.apply(init_weights_)
    
    
    def forward(self, context, context_mask, target_x):
        """ 
        Performs a forward pass through the model.
        We encode the room into the latent layer,
        then decode the latent layer to predict 
        the mean and std RIRs at the target locations. 
        """
        h = self.encoder.forward(context, context_mask)
        h = torch.stack( [h]*target_x.shape[1], dim=1) 
        h = torch.cat( (h, target_x), dim=2) 
        self.mu, self.sigma = self.decoder.forward(h)
        return self.mu, self.sigma
    

    def get_latent(self, context, context_mask):
        """
        Returns the latent representation of the context.
        A 128-dimensional vector that encodes the sound field for a room. 
        """
        h = self.encoder.forward(context, context_mask)
        return h
    
    
    def loss(self, target_y, target_mask):
        """
        The loss is the negative log-likelihood (NLL) between the the expected and actual distribution.
        The NLL can become negative, when y is real-values (Goodfellow 2018) - as is the case here. 
        """
        s = target_y.shape
        target_y = target_y.reshape( (s[0], s[1]) ) 
        mvn = Normal(self.mu, self.sigma)
        loss = -torch.sum(mvn.log_prob(target_y) * target_mask.float())
        return loss
