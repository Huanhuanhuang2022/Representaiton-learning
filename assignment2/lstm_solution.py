from pickle import NONE
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(
        self,
        vocabulary_size=40479,
        embedding_size=768,
        hidden_size=512,
        num_layers=1,
        learn_embeddings=False,
        _embedding_weight=None,
    ):

        super(LSTM, self).__init__()
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learn_embeddings = learn_embeddings

        self.embedding = nn.Embedding(
            vocabulary_size, embedding_size, padding_idx=0, _weight=_embedding_weight
        )
        self.lstm = nn.LSTM(
            embedding_size, hidden_size, num_layers=num_layers, batch_first=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, vocabulary_size, bias=False),
        )

        # Tying classifier and embedding weights (similar to GPT-1)
        self.classifier[2].weight = self.embedding.weight

        # Freeze the embedding weights, depending on learn_embeddings
        self.embedding.requires_grad_(learn_embeddings)

    def forward(self, inputs, hidden_states):
        """LSTM.

        This is a Long Short-Term Memory network for language modeling. This
        module returns for each position in the sequence the log-probabilities
        of the next token. See Lecture 05, slides 42-60.

        Parameters
        ----------
        inputs (`torch.LongTensor` of shape `(batch_size, sequence_length)`)
            The input tensor containing the token sequences.

        hidden_states (`tuple` of size 2)
            The (initial) hidden state. This is a tuple containing
            - h (`torch.FloatTensor` of shape `(num_layers, batch_size, hidden_size)`)
            - c (`torch.FloatTensor` of shape `(num_layers, batch_size, hidden_size)`)

        Returns
        -------
        log_probas (`torch.FloatTensor` of shape `(batch_size, sequence_length, vocabulary_size)`)
            A tensor containing the log-probabilities of the next token for
            all positions in each sequence of the batch. For example, `log_probas[0, 3, 6]`
            corresponds to log p(x_{5} = token_{7} | x_{0:4}) (x_{5} for the word
            after x_{4} at index 3, and token_{7} for index 6) for the 1st sequence
            of the batch (index 0).

        hidden_states (`tuple` of size 2)
            The final hidden state. This is a tuple containing
            - h (`torch.FloatTensor` of shape `(num_layers, batch_size, hidden_size)`)
            - c (`torch.FloatTensor` of shape `(num_layers, batch_size, hidden_size)`)
        """

        # ==========================
        # TODO: Write your code here
        print(inputs.shape)#batchi_size=7,sequence_length=11
        x=self.embedding(inputs)
        print(x.shape)
        print(hidden_states[0].shape)

        # Propagate input through LSTM
        out,  (hn, cn) = self.lstm(x, hidden_states) #lstm with input, hidden, and internal state
        print(f'out_lstm',hn.shape)#torch.Size([5, 7, 23])

        out = out.reshape(-1, self.hidden_size)
        print(f'out_linear',out.shape)#([5, 7, 23])

        out = self.classifier(out) #MLP layer
        print(f'out_linear',out.shape)#([5, 7, 23])
        out=out.reshape(inputs.size()[0], inputs.size()[1],-1)
        log_probas = F.log_softmax(out,dim=2)
        print(f'log_probas',log_probas.shape)#batchi_size=7,sequence_length=11,vocabulary_size

        return log_probas, (hn, cn)
        # ==========================
        pass 

    def loss(self, log_probas, targets, mask):
        """Loss function.

        This function computes the loss (negative log-likelihood).

        Parameters
        ----------
        log_probas (`torch.FloatTensor` of shape `(batch_size, sequence_length, vocabulary_size)`)
            A tensor containing the log-probabilities of the next token for
            all positions in each sequence of the batch.

        targets (`torch.LongTensor` of shape `(batch_size, sequence_length)`)
            A tensor containing the target next tokens for all positions in
            each sequence of the batch.

        mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`)
            A tensor containing values in {0, 1} only, where the value is 0
            for positions corresponding to padding in the sequence, and 1
            otherwise.

        Returns
        -------
        loss (`torch.FloatTensor` scalar)
            The scalar loss, corresponding to the (mean) negative log-likelihood.
        """

        # ==========================
        # TODO: Write your code here
        # print(f'target',targets)#7*11
        # print(f'mask',mask)#7*11        
        # # target_flat = targets.view(-1, 1)#77*1


        target_flat = targets.view(-1)
        # print(target_flat.shape)#77*1
        # print(log_probas.shape)#7*11*13
        log_probas_flat = log_probas.view(-1, log_probas.size(-1))
        print(log_probas_flat.shape)#77*13
        # log_probas_gather = -torch.gather(log_probas_flat, dim=1, index=target_flat)
        # print(f'log_probas_gather',log_probas_gather.shape)#77*1
        # target_flat=target_flat.squeeze(-1)#77

        # log_probas_gather= log_probas_gather.view(*targets.size())
        # print(log_probas_gather.shape)
        print(f'log_probas_flat',log_probas_flat)
        # print(f'log_probas_gather',log_probas_gather)
        print(f'target_flat',target_flat)

        losses = nn.NLLLoss(reduction="none")
        # loss=[]
        loss = losses(log_probas_flat, target_flat) 
        # for t in range(target_flat.size(0)):
        #     out = losses(log_probas_flat[t], target_flat[t])            
        #     # 
        #     loss.append(out)
        print(f'loss',loss)#

        # loss.backward()
        # loss = torch.stack(loss,0) 
        # loss=losses(log_probas_flat,target_flat)
            
        loss=torch.reshape(loss, (log_probas.size(0),log_probas.size(1)))
        loss=loss*mask.float()
        # print(torch.sum(loss,dim=1))
        row_mean_loss=torch.sum(loss,dim=1)/torch.sum(mask,dim=1)
        print(row_mean_loss)
        
        column_mean_loss=torch.mean(row_mean_loss)
        # print(f'loss',loss.shape)
        print(column_mean_loss)

        return column_mean_loss
        # ==========================
        pass

    def initial_states(self, batch_size, device=None):
        if device is None:
            device = next(self.parameters()).device
        shape = (self.num_layers, batch_size, self.hidden_size)

        # The initial state is a constant here, and is not a learnable parameter
        h_0 = torch.zeros(shape, dtype=torch.float, device=device)
        c_0 = torch.zeros(shape, dtype=torch.float, device=device)

        return (h_0, c_0)

    @classmethod
    def load_embeddings_from(
        cls, filename, hidden_size=512, num_layers=1, learn_embeddings=False
    ):
        # Load the token embeddings from filename
        with open(filename, "rb") as f:
            embeddings = np.load(f)
            weight = torch.from_numpy(embeddings["tokens"])

        vocabulary_size, embedding_size = weight.shape
        return cls(
            vocabulary_size,
            embedding_size,
            hidden_size,
            num_layers,
            learn_embeddings,
            _embedding_weight=weight,
        )
