import torch
import torch.nn as nn

class PELayer(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        self.device = net_params['device']
        self.pos_enc_dim = net_params.get('pos_enc_dim', 0)
        self.dataset = net_params.get('dataset', 'CYCLES')
        self.n_gape = net_params.get('n_gape', 1)
        self.gape_pooling = net_params.get('gape_pooling', 'mean')
        self.matrix_type = net_params.get('matrix_type', 'A')

        if self.n_gape > 1:
            print(f'Using {self.n_gape} automata for GAPE')

        hidden_dim = net_params['hidden_dim']

        # init initial vectors
        self.pos_initials = nn.ParameterList(
            nn.Parameter(torch.empty(self.pos_enc_dim, 1, device=self.device), requires_grad=False)
            for _ in range(self.n_gape)
        )
        for pos_initial in self.pos_initials:
            nn.init.normal_(pos_initial)

        if self.n_gape > 1:
            self.gape_pool_vec = nn.Parameter(torch.Tensor(self.n_gape, 1), requires_grad=True)
            nn.init.normal_(self.gape_pool_vec)

        # init transition weights
        self.pos_transitions = nn.ParameterList(
            nn.Parameter(torch.Tensor(self.pos_enc_dim, self.pos_enc_dim), requires_grad=False)
            for _ in range(self.n_gape)
        )
        for pos_transition in self.pos_transitions:
            nn.init.orthogonal_(pos_transition)

        # init linear layers for reshaping to hidden dim
        self.embedding_pos_encs = nn.ModuleList(nn.Linear(self.pos_enc_dim, hidden_dim) for _ in range(self.n_gape))

    def stack_strategy(self, num_nodes):
        """
            Given more than one initial weight vector, define the stack strategy.

            If n = number of nodes and k = number of weight vectors,
                by default, we repeat each initial weight vector n//k times
                and stack them together with final n-(n//k) weight vectors.
        """
        num_pos_initials = len(self.pos_initials)
        if num_pos_initials == 1:
            return torch.cat([self.pos_initials[0] for _ in range(num_nodes)], dim=1)

        remainder = num_nodes % num_pos_initials
        capacity = num_nodes - remainder
        out = torch.cat([self.pos_initials[i] for i in range(num_pos_initials)], dim=1)
        out = torch.repeat_interleave(out, capacity//num_pos_initials, dim=1)
        if remainder != 0:
            remaining_stack = torch.cat([self.pos_initials[-1] for _ in range(remainder)], dim=1)
            out = torch.cat([out, remaining_stack], dim=1)
        return out

    def forward(self, g, pos_enc=None):
        if self.n_gape == 1 and pos_enc is not None:
            return pos_enc
        else:
            pos_encs = [g.ndata[f'pos_enc_{i}'] for i in range(self.n_gape)]
            # pos_encs = [self.embedding_pos_encs[i](pos_encs[i]) for i in range(self.n_gape)]
            pos_encs = torch.stack(pos_encs, dim=0) # (n_gape, n_nodes, pos_enc_dim)
            # pos_enc_block = self.embedding_pos_enc(pos_enc_block) # (n_gape, n_nodes, hidden_dim)
            # count how many nans are in pos_enc_block
            # if self.gape_pooling == "mean":
            #     pos_enc_block = torch.mean(pos_enc_block, 0, keepdim=False) # (n_nodes, hidden_dim)
            # elif self.gape_pooling == 'sum':
            #     pos_enc_block = torch.sum(pos_enc_block, 0, keepdim=False)
            # elif self.gape_pooling == 'max':
            #     pos_enc_block = torch.max(pos_enc_block, 0, keepdim=False)[0]
            # pe = pos_enc_block
            pos_encs = pos_encs.permute(1, 2, 0) # (n_nodes, pos_enc_dim, n_gape)
            pos_encs = pos_encs @ self.gape_pool_vec
            pos_encs = pos_encs.squeeze(2)
            pe = self.embedding_pos_encs[0](pos_encs)


        return pe