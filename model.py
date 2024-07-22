import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias

        return self.act(out)


# two layers
class GCN2(nn.Module):
    def __init__(self, in_ft, out_ft, act, bias=True):
        super(GCN2, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj, du, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias

        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(out, 0)), 0)
        else:
            out = torch.bmm(adj, out)
        if self.bias is not None:
            out += self.bias

        return self.act(out)


class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            device: which device to use
        '''

        super(MLP, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
            return self.linears[self.num_layers - 1](h)

class FNN(nn.Module):
    def __init__(self, in_features, hidden, out_features, layer_num):
        super(FNN, self).__init__()
        self.linear1 = MLP(layer_num, in_features, hidden, out_features)
        self.linear2 = nn.Linear(out_features, out_features)
    def forward(self, embedding):
        x = self.linear1(embedding)
        x = self.linear2(F.relu(x))
        return x


class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq):
        return torch.mean(seq, 1)


class MaxReadout(nn.Module):
    def __init__(self):
        super(MaxReadout, self).__init__()

    def forward(self, seq):
        return torch.max(seq, 1).values


class MinReadout(nn.Module):
    def __init__(self):
        super(MinReadout, self).__init__()

    def forward(self, seq):
        return torch.min(seq, 1).values


class WSReadout(nn.Module):
    def __init__(self):
        super(WSReadout, self).__init__()

    def forward(self, seq, query):
        query = query.permute(0, 2, 1)
        sim = torch.matmul(seq, query)
        sim = F.softmax(sim, dim=1)
        sim = sim.repeat(1, 1, 64)
        out = torch.mul(seq, sim)
        out = torch.sum(out, 1)
        return out


class Discriminator(nn.Module):
    def __init__(self, n_h, negsamp_round):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

        self.negsamp_round = negsamp_round

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl):
        scs = []
        # positive
        scs.append(self.f_k(h_pl, c))

        # negative
        c_mi = c
        for _ in range(self.negsamp_round):
            c_mi = torch.cat((c_mi[-2:-1, :], c_mi[:-1, :]), 0)
            scs.append(self.f_k(h_pl, c_mi))

        logits = torch.cat(tuple(scs))

        return logits


class Decoder(nn.Module):
    def __init__(self, n_in, n_h, hidden_size = 128):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size

        self.network1 = nn.Sequential(
            nn.Linear(n_h , self.hidden_size),
            nn.PReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.PReLU(),
            nn.Linear(self.hidden_size, n_in),
            nn.PReLU()
        )
        self.network2 = nn.Sequential(
            nn.Linear(n_h * 2, self.hidden_size),
            nn.PReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.PReLU(),
            nn.Linear(self.hidden_size, n_in),
            nn.PReLU()
        )
        self.network3 = nn.Sequential(
            nn.Linear(n_h * 3, self.hidden_size),
            nn.PReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.PReLU(),
            nn.Linear(self.hidden_size, n_in),
            nn.PReLU()
        )
        self.network4 = nn.Sequential(
            nn.Linear(n_h * 4, self.hidden_size),
            nn.PReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.PReLU(),
            nn.Linear(self.hidden_size, n_in),
            nn.PReLU()
        )
        self.network5 = nn.Sequential(
            nn.Linear(n_h * 5, self.hidden_size),
            nn.PReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.PReLU(),
            nn.Linear(self.hidden_size, n_in),
            nn.PReLU()
        )
        self.network6 = nn.Sequential(
            nn.Linear(n_h * 6, self.hidden_size),
            nn.PReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.PReLU(),
            nn.Linear(self.hidden_size, n_in),
            nn.PReLU()
        )
        self.network7 = nn.Sequential(
            nn.Linear(n_h * 7, self.hidden_size),
            nn.PReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.PReLU(),
            nn.Linear(self.hidden_size, n_in),
            nn.PReLU()
        )

    def forward(self, h_raw, subgraph_size):
        sub_size = h_raw.shape[1]
        batch_size = h_raw.shape[0]
        sub_node = h_raw[:, :sub_size - 2, :]
        input_res = sub_node.reshape(batch_size, -1)
        # input_res = h_raw[:, -2, :]
        # subgraph_size = 1
        if subgraph_size == 1:
            node_recons = self.network1(input_res)
        elif subgraph_size == 2:
            node_recons = self.network2(input_res)
        elif subgraph_size == 3:
            node_recons = self.network3(input_res)
        elif subgraph_size == 4:
            node_recons = self.network4(input_res)
        elif subgraph_size == 5:
            node_recons = self.network5(input_res)
        elif subgraph_size == 6:
            node_recons = self.network6(input_res)
        elif subgraph_size == 7:
            node_recons = self.network7(input_res)
        return node_recons


class Decoder2(nn.Module):
    def __init__(self, n_in, n_h, hidden_size = 128):
        super(Decoder2, self).__init__()
        self.hidden_size = hidden_size

        self.mlp = nn.Sequential(
            nn.Linear(n_h , self.hidden_size),
            nn.PReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.PReLU(),
            nn.Linear(self.hidden_size, n_in),
            nn.PReLU()
        )

    def forward(self, h_raw_readout, subgraph_size):
        # Using readout
        node_recons = self.mlp(h_raw_readout)
        return node_recons





class Model(nn.Module):
    def  __init__(self, n_in, n_h, activation, negsamp_round, readout, have_neg = False, hidden_size = 128,
                 temperature=0.4, neg_top_k=20, graph_negsamp_round=128):
        super(Model, self).__init__()
        self.read_mode = readout
        self.hidden_size = hidden_size
        self.gcn = GCN(n_in, n_h, activation)
        # self.gcn_2layers = GCN2(n_in, n_h, activation)
        # self.gcn_same = GCN(n_in, n_h, activation)
        self.decoder = Decoder(n_in, n_h, self.hidden_size)
        # self.decoder = Decoder2(n_in, n_h, self.hidden_size)
        self.temperature = temperature
        self.have_neg = have_neg
        self.neg_top_k = neg_top_k
        self.negsamp_round = negsamp_round
        self.graph_negsamp_round = graph_negsamp_round

        if readout == 'max':
            self.read = MaxReadout()
        elif readout == 'min':
            self.read = MinReadout()
        elif readout == 'avg':
            self.read = AvgReadout()
        elif readout == 'weighted_sum':
            self.read = WSReadout()

        self.discriminator = Discriminator(n_h, negsamp_round)
        # self.degree_decoder = FNN(n_h, n_h, 1, 1)
        # self.degree_decoder = FNN(n_h, n_h, 1, 4)


    # -------------------- Inter-view / Cross-view ---------------#
    def InterViewLoss(self, h1, h2, have_neg=False, neg_top_k=20, graph_negsamp_round=127):
        h1_new = h1.clone()
        h2_new = h2.clone()
        h1_new[:, [-2, -1], :] = h1_new[:, [-1, -2], :]
        h2_new[:, [-2, -1], :] = h2_new[:, [-1, -2], :]

        h1_graph = h1_new[:, : -1, :]  # (B, subgraph_size, D)
        h1_masknode = h1_new[:,  -2: -1, :]
        h2_graph = h2_new[:, : -1, :]  # (B, subgraph_size, D)
        h2_masknode = h2_new[:,  -2: -1, :]

        if self.read_mode != 'weighted_sum':
            h1_readout = self.read(h1_graph)
            h2_readout = self.read(h2_graph)
        else:
            h1_readout = self.read(h1_graph, h1_masknode)
            h2_readout = self.read(h2_graph, h2_masknode)


        fx = lambda x: torch.exp(x / self.temperature)
        # positive pair
        sim_pos = F.cosine_similarity(h1_readout, h2_readout)
        loss_pos = fx(sim_pos)

        # negative pair
        if have_neg:
            neg_sim_list = []
            batch_size = h1_graph.shape[0]
            neg_index = list(range(batch_size))
            # for i in range((batch_size - 1)):
            for i in range(graph_negsamp_round):
                neg_index.insert(0, neg_index.pop(-1))
                perm1_readout = h1_readout[neg_index].clone()
                perm2_readout = h2_readout[neg_index].clone()
                sim_neg_1 = F.cosine_similarity(h1_readout, perm1_readout) # GCC no this item
                sim_neg_2 = F.cosine_similarity(h1_readout, perm2_readout)
                # sim_neg = (sim_neg_1 + sim_neg_2)
                # sim_neg = (sim_neg_1 + sim_neg_2) / 2
                neg_sim_list.append(torch.squeeze(sim_neg_1).detach().cpu().numpy())
                neg_sim_list.append(torch.squeeze(sim_neg_2).detach().cpu().numpy())

            neg_sim_list = torch.tensor(np.array(neg_sim_list), requires_grad=True).squeeze()
            neg_sim_list = neg_sim_list.to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

            if neg_top_k < (batch_size - 1):
                # top max k as negative pairs
                neg_sim = torch.sort(neg_sim_list, descending=False, dim=0)[0]
                sim_neg_top_k = neg_sim[:neg_top_k, :]
                loss_neg = sim_neg_top_k
                # if graph_negsamp_round > 1:
                #     loss_neg = torch.mean(loss_neg, dim=0)
                    # loss_neg = torch.sum(loss_neg, dim=0)
            else:
                loss_neg = neg_sim_list


            loss_neg = fx(loss_neg)
            if loss_neg.shape[0] > 1:
                loss_neg = torch.mean(loss_neg, dim=0)
                # loss_neg = torch.sum(loss_neg, dim=0)


            loss = -torch.log((loss_pos) / (loss_neg + loss_pos))

        else:
            loss = -torch.log(loss_pos)

        return loss, sim_pos




    def InterViewLoss_nodemask(self, h1_readout, h2_readout, have_neg = True, neg_top_k=20, graph_negsamp_round=1):
        fx = lambda x: torch.exp(x / self.temperature)
        # positive
        sim_pos = F.cosine_similarity(h1_readout, h2_readout)
        loss_pos = fx(sim_pos)

        # negative pair
        if have_neg:
            neg_sim_list = []
            batch_size = h1_readout.shape[0]
            neg_index = list(range(batch_size))
            # for i in range((batch_size - 1)):
            for i in range(graph_negsamp_round):
                neg_index.insert(0, neg_index.pop(-1))
                perm1_readout = h1_readout[neg_index].clone()
                perm2_readout = h2_readout[neg_index].clone()
                sim_neg_1 = F.cosine_similarity(h1_readout, perm1_readout)
                sim_neg_2 = F.cosine_similarity(h1_readout, perm2_readout)
                # sim_neg = (sim_neg_1 + sim_neg_2)
                sim_neg = (sim_neg_1 + sim_neg_2) / 2
                neg_sim_list.append(torch.squeeze(sim_neg).detach().cpu().numpy())

            neg_sim_list = torch.tensor(np.array(neg_sim_list), requires_grad=True).squeeze()
            neg_sim_list = neg_sim_list.to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

            if neg_top_k < (batch_size - 1):
                neg_sim = torch.sort(neg_sim_list, descending=False, dim=0)[0]
                sim_neg_top_k = neg_sim[:neg_top_k, :]
                loss_neg = sim_neg_top_k
            else:
                loss_neg = neg_sim_list

            if graph_negsamp_round > 1:
                loss_neg = torch.mean(loss_neg, dim=0)
                # loss_neg = torch.sum(loss_neg, dim=0)

            loss_neg = fx(loss_neg)

            loss = -torch.log((loss_pos) / (loss_neg + loss_pos))

        else:
            loss = -torch.log(loss_pos)

        return loss


    def InterViewLoss_mix(self, h1, h2, have_neg=False, neg_top_k=20, graph_negsamp_round=1):
    # h1 without masknode; h2 with masknode
        fx = lambda x: torch.exp(x / self.temperature)

        h1_new = h1.clone()
        h1_new[:, [-2, -1], :] = h1_new[:, [-1, -2], :]

        h1_graph = h1_new[:, : -1, :]  # (B, subgraph_size, D)
        h1_masknode = h1_new[:, -2: -1, :]
        h2_graph = h2[:, : -1, :]  # (B, subgraph_size, D)
        h2_masknode = h2[:, -2: -1, :]

        if self.read_mode != 'weighted_sum':
            h1_readout = self.read(h1_graph)
            h2_readout = self.read(h2_graph)
        else:
            h1_readout = self.read(h1_graph, h1_masknode)
            h2_readout = self.read(h2_graph, h2_masknode)

        # positive pair
        sim_pos = F.cosine_similarity(h1_readout, h2_readout)
        loss_pos = fx(sim_pos)

        # negative pair
        if have_neg:
            neg_sim_list = []
            batch_size = h1_readout.shape[0]
            neg_index = list(range(batch_size))
            # for i in range((batch_size - 1)):
            for i in range(graph_negsamp_round):
                neg_index.insert(0, neg_index.pop(-1))
                perm1_readout = h1_readout[neg_index].clone()
                perm2_readout = h2_readout[neg_index].clone()
                sim_neg_1 = F.cosine_similarity(h1_readout, perm1_readout)
                sim_neg_2 = F.cosine_similarity(h1_readout, perm2_readout)
                # sim_neg = (sim_neg_1 + sim_neg_2)
                sim_neg = (sim_neg_1 + sim_neg_2) / 2
                neg_sim_list.append(torch.squeeze(sim_neg).detach().cpu().numpy())

            neg_sim_list = torch.tensor(np.array(neg_sim_list), requires_grad=True).squeeze()
            neg_sim_list = neg_sim_list.to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

            if neg_top_k < (batch_size - 1):
                neg_sim = torch.sort(neg_sim_list, descending=False, dim=0)[0]
                sim_neg_top_k = neg_sim[:neg_top_k, :]
                loss_neg = sim_neg_top_k
            else:
                loss_neg = neg_sim_list

            if graph_negsamp_round > 1:
                loss_neg = torch.mean(loss_neg, dim=0)
                # loss_neg = torch.sum(loss_neg, dim=0)

            loss_neg = fx(loss_neg)

            loss = -torch.log((loss_pos) / (loss_neg + loss_pos))

        else:
            loss = -torch.log(loss_pos)

        return loss


    def InterViewLoss_mix2(self, h1, h2, have_neg=False, neg_top_k=20, graph_negsamp_round=1):
    # h1 without masknode; h2 with masknode
        fx = lambda x: torch.exp(x / self.temperature)

        h1_new = h1.clone()
        h1_new[:, [-2, -1], :] = h1_new[:, [-1, -2], :]

        h1_graph = h1_new[:, : -1, :]  # (B, subgraph_size, D)
        h1_masknode = h1_new[:, -2: -1, :]
        h2_graph = h2[:, : -2, :]  # (B, subgraph_size, D)
        h2_masknode = h2[:, -2: -1, :]

        if self.read_mode != 'weighted_sum':
            h1_readout = self.read(h1_graph)
            h2_readout = self.read(h2_graph)
        else:
            h1_readout = self.read(h1_graph, h1_masknode)
            h2_readout = self.read(h2_graph, h2_masknode)

        # positive pair
        sim_pos = F.cosine_similarity(h1_readout, h2_readout)
        loss_pos = fx(sim_pos)

        # negative pair
        if have_neg:
            neg_sim_list = []
            batch_size = h1_readout.shape[0]
            neg_index = list(range(batch_size))
            # for i in range((batch_size - 1)):
            for i in range(graph_negsamp_round):
                neg_index.insert(0, neg_index.pop(-1))
                perm1_readout = h1_readout[neg_index].clone()
                perm2_readout = h2_readout[neg_index].clone()
                sim_neg_1 = F.cosine_similarity(h1_readout, perm1_readout)
                sim_neg_2 = F.cosine_similarity(h1_readout, perm2_readout)
                # sim_neg = (sim_neg_1 + sim_neg_2)
                sim_neg = (sim_neg_1 + sim_neg_2) / 2
                neg_sim_list.append(torch.squeeze(sim_neg).detach().cpu().numpy())

            neg_sim_list = torch.tensor(np.array(neg_sim_list), requires_grad=True).squeeze()
            neg_sim_list = neg_sim_list.to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

            if neg_top_k < (batch_size - 1):
                neg_sim = torch.sort(neg_sim_list, descending=False, dim=0)[0]
                sim_neg_top_k = neg_sim[:neg_top_k, :]
                loss_neg = sim_neg_top_k
            else:
                loss_neg = neg_sim_list

            if graph_negsamp_round > 1:
                loss_neg = torch.mean(loss_neg, dim=0)
                # loss_neg = torch.sum(loss_neg, dim=0)

            loss_neg = fx(loss_neg)

            loss = -torch.log((loss_pos) / (loss_neg + loss_pos))

        else:
            loss = -torch.log(loss_pos)

        return loss




    def degree_decoding(self, node_embeddings):
        x = self.degree_decoder(node_embeddings)
        degree_logits = F.relu(x)
        return degree_logits



    def forward(self, feature1, adj1, raw1, size1, feature2, adj2, raw2, size2, sparse=False):
        # h_raw_1 = self.gcn(raw1, adj1, sparse)
        # h_raw_2 = self.gcn_2layers(raw2, adj2, sparse)
        # h1 = self.gcn(feature1, adj1, sparse)
        # h2 = self.gcn_2layers(feature2, adj2, sparse)
        #-----------------------------------#
        h_raw_1 = self.gcn(raw1, adj1, sparse)
        h_raw_2 = self.gcn(raw2, adj2, sparse)
        h1 = self.gcn(feature1, adj1, sparse)
        h2 = self.gcn(feature2, adj2, sparse)
        # -----------------------------------#
        # h_raw_1 = self.gcn(raw1, adj1, sparse)
        # h_raw_2 = self.gcn_same(raw2, adj2, sparse)
        # h1 = self.gcn(feature1, adj1, sparse)
        # h2 = self.gcn_same(feature2, adj2, sparse)
        # -----------------------------------#



        # --------------------Node Reconstruction loss---------------#
        node_recons_1 = self.decoder(h_raw_1, size1)
        node_recons_2 = self.decoder(h_raw_2, size2)

        # --------------------With mask node readout Reconstruction loss---------------#
        # h1_raw_read = self.read(h_raw_1[:, : -2, :])
        # h2_raw_read = self.read(h_raw_2[:, : -2, :])
        # node_recons_1 = self.decoder(h1_raw_read, size1)
        # node_recons_2 = self.decoder(h2_raw_read, size2)

        # node_recons_1 = self.decoder(h_raw_1[:, -2 : -1, :].squeeze(), size1)
        # node_recons_2 = self.decoder(h_raw_1[:, -2 : -1, :].squeeze(), size2)

        # --------------------Degree Reconstruction loss---------------#
        # degree_logits_1 = self.degree_decoding(h_raw_1[:,-2,:]) #[:,-1,:] 用非mask点； [:,-2,:] 用mask点
        # degree_logits_2 = self.degree_decoding(h_raw_2[:,-2,:])
        # degree_logits_1 = torch.squeeze(degree_logits_1)
        # degree_logits_2 = torch.squeeze(degree_logits_2)

        # --------------------Intra-view / CoLa loss---------------#
        if self.read_mode != 'weighted_sum':
            h_node_1 = h1[:, -1, :]
            h_graph_read_1 = self.read(h1[:, : -1, :])
            h_node_2 = h2[:, -1, :]
            h_graph_read_2 = self.read(h2[:, : -1, :])
        else:
            h_node_1 = h1[:, -1, :]
            h_graph_read_1 = self.read(h1[:, : -1, :], h1[:, -2: -1, :])
            h_node_2 = h2[:, -1, :]
            h_graph_read_2 = self.read(h2[:, : -1, :], h2[:, -2: -1, :])

        cola_disc_1 = self.discriminator(h_graph_read_1, h_node_1)
        cola_disc_2 = self.discriminator(h_graph_read_2, h_node_2)

        inter_loss_1, sim_pos_1 = self.InterViewLoss(h1, h2, self.have_neg, self.neg_top_k, self.graph_negsamp_round)
        inter_loss_2, sim_pos_2 = self.InterViewLoss(h2, h1, self.have_neg, self.neg_top_k, self.graph_negsamp_round)
        # inter_loss_1 = self.InterViewLoss_nodemask(h_graph_read_1, h_graph_read_2, self.have_neg, self.neg_top_k)
        # inter_loss_2 = self.InterViewLoss_nodemask(h_graph_read_2, h_graph_read_1, self.have_neg, self.neg_top_k)
        # inter_loss_1 = self.InterViewLoss_mix2(h1, h2, self.have_neg, self.neg_top_k)
        # inter_loss_2 = self.InterViewLoss_mix2(h2, h1, self.have_neg, self.neg_top_k)
        inter_loss = (inter_loss_1 + inter_loss_2) / 2
        sim_pos = (sim_pos_1 + sim_pos_2) / 2

        return node_recons_1, node_recons_2, cola_disc_1, cola_disc_2, inter_loss, sim_pos
