import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import activation_getter
from forget_mult import ForgetMult



class Model2(nn.Module):
    """
    Convolutional Sequence Embedding Recommendation Model (Caser)[1].

    [1] Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding, Jiaxi Tang and Ke Wang , WSDM '18

    Parameters
    ----------

    num_users: int,
        Number of users.
    num_items: int,
        Number of items.
    model_args: args,
        Model-related arguments, like latent dimensions.
    """

    def __init__(self, num_users, num_items, model_args):
        super(Model2, self).__init__()
        self.args = model_args

        # init args
        m=self.args.d
        L = self.args.L
        dims = self.args.d
        self.n_l = self.args.n_l
        self.n_h = self.args.nh
        self.n_v = self.args.nv
        self.drop_ratio = self.args.drop
        self.ac_conv = activation_getter[self.args.ac_conv]
        self.ac_fc = activation_getter[self.args.ac_fc]
        self.conv_gate = activation_getter['sigm']

        # user and item embeddings
        self.user_embeddings = nn.Embedding(num_users, dims)
        self.item_embeddings = nn.Embedding(num_items, dims)

        

        # horizontal conv layer
        lengths = [i + 1 for i in range(L)]
        self.conv_h = nn.ModuleList([nn.Conv2d(1, m, (i, dims)) for i in lengths])

        # fully-connected layer
        self.fc1_dim_v = self.n_v * dims
        self.fc1_dim_h = self.n_h * len(lengths)
        fc1_dim_in = self.fc1_dim_v + self.fc1_dim_h
        # W1, b1 can be encoded with nn.Linear
        self.fc1 = nn.Linear(dims+dims, dims)
        # W2, b2 are encoded with nn.Embedding, as we don't need to compute scores for all items
        self.W2 = nn.Embedding(num_items, dims)
        self.b2 = nn.Embedding(num_items, 1)

        # dropout
        self.dropout = nn.Dropout(self.drop_ratio)

        # weight initialization
        self.user_embeddings.weight.data.normal_(0, 1.0 / self.user_embeddings.embedding_dim)
        self.item_embeddings.weight.data.normal_(0, 1.0 / self.item_embeddings.embedding_dim)
        self.W2.weight.data.normal_(0, 1.0 / self.W2.embedding_dim)
        self.b2.weight.data.zero_()

        self.cache_x = None

    def forward(self, seq_var, user_var, item_var, for_pred=False):
        """
        The forward propagation used to get recommendation scores, given
        triplet (user, sequence, targets).

        Parameters
        ----------

        seq_var: torch.FloatTensor with size [batch_size, max_sequence_length]
            a batch of sequence
        user_var: torch.LongTensor with size [batch_size]
            a batch of user
        item_var: torch.LongTensor with size [batch_size]
            a batch of items
        for_pred: boolean, optional
            Train or Prediction. Set to True when evaluation.
        """

        # Embedding Look-up
        item_embs = self.item_embeddings(seq_var).unsqueeze(1)  # use unsqueeze() to get 4-D
        user_emb = self.user_embeddings(user_var).squeeze(1)

        # Convolutional Layers
        out, out_h, out_v = None, None, None
        
        # horizontal conv layer
        out_hs = list()
        o = torch.zeros(list(item_embs.size())[0],self.args.d)
        for j in range(self.args.L):
            conv=self.conv_h[j]
            # for i in range(self.args.d):
             
            paddedemb =  torch.zeros(list(item_embs.size())[0],1,j+list(item_embs.size())[2],list(item_embs.size())[3])
            paddedemb[:,:,j:,:]=item_embs[:,:,:,:]
            conv_out = self.ac_conv(conv(paddedemb).squeeze(3))
            out_hs.append(conv_out)
            out_h = self.conv_gate(conv_out) #torch.cat(out_hs, 2)  # prepare for fully connect

            f = out_h.permute(2,0,1)
            x=item_embs.squeeze(1).permute(1,0,2)
            # for i in range()
            xprev = x
            for i in range(self.n_l):
                h = ForgetMult()(f,xprev,None,False)
                xprev = h

            hw = torch.sum(h,dim=0)
            o += hw



        # Fully-connected Layers
        
        x = torch.cat([o, user_emb], 1)
        
        # apply dropout
        out = self.dropout(x)

        # fully-connected layer
        z = (self.fc1(out))

        w2 = self.W2(item_var)
        b2 = self.b2(item_var)

        if for_pred:
            w2 = w2.squeeze()
            b2 = b2.squeeze()
            res = (z * w2).sum(1) + b2
        else:
            res = torch.baddbmm(b2, w2, z.unsqueeze(2)).squeeze()

        return res
