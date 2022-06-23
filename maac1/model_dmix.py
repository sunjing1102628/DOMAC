# File name: model.py
# Author: Zachry
# Time: 2019-12-04
# Description: The model structure of QMIX
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

'''class Q_Network(nn.Module):
    def __init__(self, obs_size, act_size, args,seed):
        super(Q_Network, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.hidden_size = args.q_net_hidden_size
       # self.mlp_in_layer = nn.Linear(obs_size+act_size, args.q_net_out[0])
        self.mlp_in_layer = nn.Linear(obs_size, args.q_net_out[0])
        self.mlp_out_layer = nn.Linear(args.q_net_hidden_size, act_size)
        self.GRU_layer = nn.GRUCell(args.q_net_out[0], args.q_net_hidden_size)
        self.ReLU = nn.ReLU()


        self.train()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.mlp_in_layer.weight)
        nn.init.xavier_uniform_(self.mlp_out_layer.weight)
    
    def init_hidden(self, args):
        return self.mlp_in_layer.weight.new(1, args.q_net_hidden_size).zero_()

    def forward(self, obs_a_cat, hidden_last):
        #x = self.ReLU(self.mlp_in_layer(obs_a_cat))
        x = self.ReLU(self.mlp_in_layer(obs_a_cat))
        gru_out = self.GRU_layer(x, hidden_last)
        output = self.mlp_out_layer(gru_out)
        return output, gru_out'''
class IQNRNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(IQNRNNAgent, self).__init__()
        self.args = args

        self.quantile_embed_dim = args.quantile_embed_dim #64

        self.n_quantiles = args.n_quantiles #n_quantiles: 8 # N in paper
        self.n_target_quantiles = args.n_target_quantiles #n_target_quantiles: 8 # N' in paper
        self.n_approx_quantiles = args.n_approx_quantiles #n_approx_quantiles: 64 # \hat{N} in paper, for approximating Q
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.phi = nn.Linear(args.quantile_embed_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state, forward_type="policy"):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        if forward_type == "approx":
            n_rnd_quantiles = self.n_approx_quantiles
        elif forward_type == "policy":
            n_rnd_quantiles = self.n_quantiles
        elif forward_type == "target":
            n_rnd_quantiles = self.n_target_quantiles
        else:
            raise ValueError("Unknown forward_type")
        shape = h.shape
        batch_size = shape[0]
        h2 = h.reshape(batch_size, 1, self.args.rnn_hidden_dim).expand(-1, n_rnd_quantiles, -1).reshape(-1, self.args.rnn_hidden_dim)
        assert h2.shape == (batch_size * n_rnd_quantiles, self.args.rnn_hidden_dim)
        shape = h2.shape
        # Generate random quantiles
        if self.args.name == "diql":
            rnd_quantiles = torch.rand(batch_size * n_rnd_quantiles).cuda()
            batch_size_grouped = batch_size
        else:
            # Same quantiles for optimizing quantile mixture
            batch_size_grouped = batch_size // self.args.n_agents #batch_size_grouped
            #rnd_quantiles = th.rand(batch_size_grouped, 1, n_rnd_quantiles).cuda()
            rnd_quantiles = torch.rand(batch_size_grouped, 1, n_rnd_quantiles) #rnd_quantiles torch.Size([1, 1, 32])

            rnd_quantiles = rnd_quantiles.reshape(-1) #torch.Size([32])

        assert rnd_quantiles.shape == (batch_size_grouped * n_rnd_quantiles,)
        # Expand quantiles to cosine features
        quantiles = rnd_quantiles.view(batch_size_grouped * n_rnd_quantiles, 1).expand(-1, self.quantile_embed_dim)
        assert quantiles.shape == (batch_size_grouped * n_rnd_quantiles, self.quantile_embed_dim)
        #feature_id = th.arange(0, self.quantile_embed_dim).cuda()
        feature_id = torch.arange(0, self.quantile_embed_dim) #[0, ..., 63]

        feature_id = feature_id.view(1, -1).expand(batch_size_grouped * n_rnd_quantiles, -1) #torch.Size([32, 64])

        assert feature_id.shape == (batch_size_grouped * n_rnd_quantiles, self.quantile_embed_dim)
        cos = torch.cos(math.pi * feature_id * quantiles) #torch.Size([32, 64])

        assert cos.shape == (batch_size_grouped * n_rnd_quantiles, self.quantile_embed_dim)
        # Quantile embedding network (phi)
        q_phi = F.relu(self.phi(cos))
        assert q_phi.shape == (batch_size_grouped * n_rnd_quantiles, self.args.rnn_hidden_dim)
        if self.args.name != "diql":
            q_phi = q_phi.view(batch_size_grouped, n_rnd_quantiles, self.args.rnn_hidden_dim)
            q_phi = q_phi.unsqueeze(1).expand(-1, self.args.n_agents, -1, -1).contiguous().view(-1, self.args.rnn_hidden_dim)
        assert q_phi.shape == (batch_size * n_rnd_quantiles, self.args.rnn_hidden_dim)
        q_vals = self.fc2(h2 * q_phi)
        q_vals = q_vals.view(-1, n_rnd_quantiles, self.args.n_actions)
        assert q_vals.shape == (batch_size, n_rnd_quantiles, self.args.n_actions)
        q_vals = q_vals.permute(0, 2, 1) #torch.Size([2, 6, 32])
        assert q_vals.shape == (batch_size, self.args.n_actions, n_rnd_quantiles)
        rnd_quantiles = rnd_quantiles.view(batch_size_grouped, n_rnd_quantiles) #torch.Size([1, 32])
        return q_vals, h, rnd_quantiles

    
class Hyper_Network(nn.Module):
    def __init__(self, shape_state, shape_hyper_net, args,seed):
        super(Hyper_Network, self).__init__()
        self.hyper_net_pars = shape_hyper_net
        self.seed = torch.manual_seed(seed)
        self.w1_layer = nn.Linear(shape_state, shape_hyper_net['w1_size'])
        self.w2_layer = nn.Linear(shape_state, shape_hyper_net['w2_size'])
        self.b1_layer = nn.Linear(shape_state, shape_hyper_net['b1_size'])
        self.b2_layer_i = nn.Linear(shape_state, args.shape_hyper_b2_hidden)
        self.b2_layer_h = nn.Linear(args.shape_hyper_b2_hidden, shape_hyper_net['b2_size'])
        self.LReLU = nn.LeakyReLU(0.01)
        self.ReLU = nn.ReLU()

        #self.reset_parameters()
        self.train()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w1_layer.weight)
        nn.init.xavier_uniform_(self.w2_layer.weight)
        nn.init.xavier_uniform_(self.b1_layer.weight)
        nn.init.xavier_uniform_(self.b2_layer_i.weight)
        nn.init.xavier_uniform_(self.b2_layer_h.weight)

    def forward(self, state):
        w1_shape = self.hyper_net_pars['w1_shape']
        w2_shape = self.hyper_net_pars['w2_shape']
        w1 = torch.abs(self.w1_layer(state)).view(-1, w1_shape[0], w1_shape[1])
        w2 = torch.abs(self.w2_layer(state)).view(-1, w2_shape[0], w2_shape[1])
        b1 = self.b1_layer(state).view(-1, 1, self.hyper_net_pars['b1_shape'][0])
        #x = self.LReLU(self.b2_layer_i(state))
        x = self.ReLU(self.b2_layer_i(state))
        b2 = self.b2_layer_h(x).view(-1, 1, self.hyper_net_pars['b2_shape'][0])
        return {'w1':w1, 'b1':b1, 'w2':w2, 'b2':b2}
        
'''class Mixing_Network(nn.Module):
    def __init__(self, action_size, num_agents, args,seed):
        super(Mixing_Network, self).__init__()
        # action_size * num_agents = the num of Q values
        self.w1_shape = torch.Size((num_agents, args.mix_net_out[0]))
        self.b1_shape = torch.Size((args.mix_net_out[0], ))
        self.w2_shape = torch.Size((args.mix_net_out[0], args.mix_net_out[1]))
        self.b2_shape = torch.Size((args.mix_net_out[1], ))
        self.w1_size = self.w1_shape[0] * self.w1_shape[1]
        self.b1_size = self.b1_shape[0]
        self.w2_size = self.w2_shape[0] * self.w2_shape[1]
        self.b2_size = self.b2_shape[0]
        self.pars = {'w1_shape':self.w1_shape, 'w1_size':self.w1_size, \
                'w2_shape':self.w2_shape, 'w2_size':self.w2_size, \
                'b1_shape':self.b1_shape, 'b1_size':self.b1_size, \
                'b2_shape':self.b2_shape, 'b2_size':self.b2_size, }
        self.LReLU = nn.LeakyReLU(0.001)
        self.ReLU = nn.ReLU()
        self.seed = torch.manual_seed(seed)
    
    def forward(self, q_values, hyper_pars):
        x = self.ReLU(torch.bmm(q_values, hyper_pars['w1']) + hyper_pars['b1'])
        output = torch.bmm(x, hyper_pars['w2']) + hyper_pars['b2']
        return output.view(-1)'''
class DMixer(nn.Module):
    def __init__(self, args):
        super(DMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))

        self.embed_dim = args.mixing_embed_dim
        self.hypernet_embed = args.hypernet_embed

        self.n_quantiles = args.n_quantiles
        self.n_target_quantiles = args.n_target_quantiles
        self.n_agents = args.n_agents

        if getattr(args, "hypernet_layers", 1) == 1:
            self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
            self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)
        elif getattr(args, "hypernet_layers", 1) == 2:
            hypernet_embed = self.hypernet_embed
            self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                        nn.ReLU(),
                                        nn.Linear(hypernet_embed, self.embed_dim * self.n_agents))
            self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                        nn.ReLU(),
                                        nn.Linear(hypernet_embed, self.embed_dim))
        elif getattr(args, "hypernet_layers", 1) > 2:
            raise Exception("Sorry >2 hypernet layers is not implemented!")
        else:
            raise Exception("Error setting number of hypernet layers.")

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                            nn.ReLU(),
                            nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, states, target):
        batch_size = agent_qs.shape[0]
        episode_length = agent_qs.shape[1]
        if target:
            n_rnd_quantiles = self.n_target_quantiles
        else:
            n_rnd_quantiles = self.n_quantiles
        assert agent_qs.shape == (batch_size, episode_length, self.n_agents, n_rnd_quantiles)
        q_mixture = agent_qs.sum(dim=2, keepdim=True)
        assert q_mixture.shape == (batch_size, episode_length, 1, n_rnd_quantiles)
        q_vals_expected = agent_qs.mean(dim=3, keepdim=True)
        q_vals_sum = q_vals_expected.sum(dim=2, keepdim=True)
        assert q_vals_expected.shape == (batch_size, episode_length, self.n_agents, 1)
        assert q_vals_sum.shape == (batch_size, episode_length, 1, 1)

        # Factorization network
        q_joint_expected = self.forward_qmix(q_vals_expected, states)
        assert q_joint_expected.shape == (batch_size, episode_length, 1, 1)

        # Shape network
        q_vals_sum = q_vals_sum.expand(-1, -1, -1, n_rnd_quantiles)
        q_joint_expected = q_joint_expected.expand(-1, -1, -1, n_rnd_quantiles)
        assert q_vals_sum.shape == (batch_size, episode_length, 1, n_rnd_quantiles)
        assert q_joint_expected.shape == (batch_size, episode_length, 1, n_rnd_quantiles)
        q_joint = q_mixture - q_vals_sum + q_joint_expected
        assert q_joint.shape == (batch_size, episode_length, 1, n_rnd_quantiles)
        return q_joint

    def forward_qmix(self, agent_qs, states):
        batch_size = agent_qs.shape[0]
        episode_length = agent_qs.shape[1]
        assert agent_qs.shape == (batch_size, episode_length, self.n_agents, 1)
        bs = agent_qs.size(0)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)
        assert agent_qs.shape == (batch_size * episode_length, 1, self.n_agents)
        assert states.shape == (batch_size, episode_length, self.state_dim)
        states = states.reshape(-1, self.state_dim)
        assert states.shape == (batch_size * episode_length, self.state_dim)
        # First layer
        w1 = th.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(th.bmm(agent_qs, w1) + b1)
        # Second layer
        w_final = torch.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)
        # Compute final output
        y = torch.bmm(hidden, w_final) + v
        # Reshape and return
        q_tot = y.view(bs, -1, 1)
        assert q_tot.shape == (batch_size, episode_length, 1)
        q_tot = q_tot.unsqueeze(3)
        assert q_tot.shape == (batch_size, episode_length, 1, 1)
        return q_tot

