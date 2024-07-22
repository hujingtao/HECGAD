from model import *
from utils import *
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc
from sklearn.preprocessing import MinMaxScaler

import random
import os
import dgl
import torch
import argparse
from tqdm import tqdm



os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Set argument
parser = argparse.ArgumentParser(description='ANIMATE')
parser.add_argument('--dataset', type=str, default='cora')  # 'BlogCatalog'  'Flickr'  'ACM'  'cora'  'citeseer'  'pubmed'
parser.add_argument('--lr', type=float)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--seed', type=int, default=3)
parser.add_argument('--embedding_dim', type=int, default=64)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--drop_prob', type=float, default=0.0)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--readout', type=str, default='avg')  # max min avg  weighted_sum
parser.add_argument('--auc_test_rounds', type=int, default=256)
parser.add_argument('--negsamp_ratio', type=int, default=1)
parser.add_argument('--hidden_size', type=int, default=64)
parser.add_argument('--beta_recon', type=float)
parser.add_argument('--alpha_inter', type=float)
parser.add_argument('--alpha_intra', type=float, default=1)
parser.add_argument('--temperature', type=float, default=3, help='temperature for fx')
parser.add_argument('--have_neg', type=bool, help='anomaly score and LOSS contain negtive pairs OT', default=True)

parser.add_argument('--graph_negsamp_round', type=int, default=127)
parser.add_argument('--neg_top_k', type=int, default=10)

parser.add_argument('--subgraph_size_1', type=int, help='view 1')
parser.add_argument('--subgraph_size_2', type=int, help='view 2')
parser.add_argument('--restart_prob_1', type=float, help='RWR restart probability on view 1', default=0.9)
parser.add_argument('--restart_prob_2', type=float, help='RWR restart probability on view 2', default=0.1)
parser.add_argument('--subgraph_mode', type=str, default='1+near')  # 1/2+1/2  1+1   1+1connect2   1+near:view 2 is related to view1
parser.add_argument('--patience', type=int, default=40, help='control early stop')

# parser.add_argument('--subgraph_size', type=int, default=4)

args = parser.parse_args()


# args.graph_negsamp_round = 127
# for args.alpha_inter in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1]:
# for args.beta_degree in [0, 0.0001, 0.001, 0.01, 0.1, 0.5, 1]:
# for args.subgraph_mode in ['1+1', '2+2', '1+1connect2', 'random']:
# for args.neg_top_k in [1, 5, 10, 20, 30, 40, 50, 60, 80, 100, 128]:
# for args.subgraph_size_1 in [2,4,6,8]:
# for args.subgraph_size_2 in [2,4,6,8]:
# for args.beta_recon in [0.2, 0.4, 0.6, 0.8]:
# for args.alpha_inter in [0.2, 0.4, 0.6, 0.8]:
# for args.seed in [3,1006,2023,3]:
# cnt_num = 0
# for cnt_num in range(5):
#     cnt_num += 1
# for args.subgraph_size_1 in [2, 4, 6]:
#     for args.subgraph_size_2 in [4, 6, 8]:
#         if args.subgraph_size_2 <= args.subgraph_size_1:
#             continue
# for args.alpha_inter in [0, 1]:
#     for args.beta_recon in [0, 1]:





# for args.dataset in ['cora','citeseer','BlogCatalog','ACM','pubmed','Flickr','Disney', 'Books', 'Reddit']:




# if args.lr is None:
if args.dataset in ['cora', 'citeseer', 'pubmed']:
    args.lr = 2e-3
elif args.dataset in ['BlogCatalog', 'Disney']:
    args.lr = 1e-2
elif args.dataset in ['ACM', 'acmv9']:
    args.lr = 5e-3
elif args.dataset == 'Flickr':
    args.lr = 1e-3
elif args.dataset in ['Books', 'citation']:
    args.lr = 3e-3
elif args.dataset in ['Reddit']:
    args.lr = 1e-1


# if args.alpha_inter == 1:
if args.dataset in ['citeseer', 'ACM', 'acmv9']:
    args.alpha_inter = 0.2
elif args.dataset in ['cora', 'Flickr', 'Reddit', 'Books','pubmed']:
    args.alpha_inter = 0.4
elif args.dataset in ['BlogCatalog', 'Disney']:
    args.alpha_inter = 0.8
else:
    args.alpha_inter = 0.4


# if args.beta_recon == 1:
if args.dataset in ['ACM', 'Disney', 'acmv9']:
    args.beta_recon = 0.2
elif args.dataset in ['cora']:
    args.beta_recon = 0.4
elif args.dataset in ['citeseer','pubmed']:
    args.beta_recon = 0.6
elif args.dataset in ['BlogCatalog', 'Flickr', 'Books', 'Reddit']:
    args.beta_recon = 0.8
else:
    args.beta_recon = 0.8



if args.dataset in ['Books', 'Disney']:
    args.epochs = 50


# if args.subgraph_size_1 is None:
if args.dataset in ['cora', 'citeseer', 'ACM']:
    args.subgraph_size_1 = 2
elif args.dataset in ['Flickr', 'pubmed']:
    args.subgraph_size_1 = 4
elif args.dataset in ['BlogCatalog', 'Disney', 'Reddit', 'Books']:
    args.subgraph_size_1 = 6
else:
    args.subgraph_size_1 = 2

# if args.subgraph_size_2 is None:
if args.dataset in ['cora']:
    args.subgraph_size_2 = 4
elif args.dataset in ['Flickr', 'ACM']:
    args.subgraph_size_2 = 6
elif args.dataset in ['citeseer', 'BlogCatalog', 'Disney', 'Reddit', 'Books', 'pubmed']:
    args.subgraph_size_2 = 8
else:
    args.subgraph_size_2 = 8






# config = yaml.load(open('config.yaml'), Loader=SafeLoader)[args.dataset]
# # combine args and config
# for k, v in config.items():
#     args.__setattr__(k, v)

AUC_list = []

dataset = args.dataset
beta_recon = args.beta_recon
alpha_inter = args.alpha_inter
alpha_intra = args.alpha_intra
batch_size = args.batch_size
subgraph_size_1 = args.subgraph_size_1
subgraph_size_2 = args.subgraph_size_2

seed = args.seed
temperature = args.temperature
have_neg = args.have_neg
lr = args.lr
epochs = args.epochs
graph_negsamp_round = args.graph_negsamp_round

print('Dataset: ', dataset)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# seed everything
dgl.random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['OMP_NUM_THREADS'] = '1'
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
torch.use_deterministic_algorithms(True)


# Load and preprocess data
if dataset in ['cora','citeseer','BlogCatalog','ACM', 'pubmed','Flickr','citation', 'uai', 'uat', 'acmv9']:
    adj, features, ano_label = load_mat(dataset)
elif dataset in ['Disney', 'Books', 'Reddit', 'Weibo', 'Enron']:
    adj, features, ano_label = load_pt(dataset)
# elif dataset in ['Yelp', 'Elliptic']:
#     adj, features, ano_label = load_dat(dataset)
else:
    raise ValueError("Dataset file not provided!")
A = adj
node_num = A.shape[0]
degree = np.array(np.sum(A, axis=0)).squeeze()
total_edge = np.sum(A)
avg_degree = total_edge / node_num
#############################################
# batch_size = node_num
###########################################33
dgl_graph = adj_to_dgl_graph(adj)
raw_feature = features.todense()
features, _ = preprocess_features(features)
nb_nodes = features.shape[0]
ft_size = features.shape[1]
adj = normalize_adj(adj)
adj = (adj + sp.eye(adj.shape[0])).todense()
b_adj = adj

features = torch.FloatTensor(features[np.newaxis])
raw_feature = torch.FloatTensor(raw_feature[np.newaxis])

adj = torch.FloatTensor(adj[np.newaxis])
b_adj = torch.FloatTensor(b_adj[np.newaxis])

degree = torch.FloatTensor(degree)

# Initialize model and optimiser
model = Model(n_in=ft_size, n_h=args.embedding_dim, activation='prelu', negsamp_round=args.negsamp_ratio,
              readout=args.readout, hidden_size=args.hidden_size, temperature=args.temperature,
              have_neg=have_neg, neg_top_k=args.neg_top_k, graph_negsamp_round=args.graph_negsamp_round)
optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)

if torch.cuda.is_available():
    print('Using CUDA')
    degree = degree.to(device)
    model.to(device)
    features = features.to(device)
    raw_feature = raw_feature.to(device)
    adj = adj.to(device)
    b_adj = b_adj.to(device)
    b_xent = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio]).to(device))
else:
    b_xent = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio]))
xent = nn.CrossEntropyLoss()
cnt_wait = 0
best = 1e9
best_t = 0
mse_loss = nn.MSELoss(reduction='mean')
degree_loss_func = nn.MSELoss()

# HU
if nb_nodes % batch_size == 0:
    batch_num = nb_nodes // batch_size
else:
    batch_num = nb_nodes // batch_size + 1
# sample half
# batch_num = int((nb_nodes / batch_size) // 2)



# # Train model
with tqdm(total=epochs) as pbar:
    pbar.set_description('Training')

    for epoch in range(epochs):

        model.train()

        all_idx = list(range(nb_nodes))

        # random.Random(seed).shuffle(all_idx)
        random.shuffle(all_idx)

        # loss_list = []
        # loss_list_1 = []
        # loss_list_2 = []
        # loss_list_3 = []
        # p = 0
        loss_1 = 0.
        loss_2 = 0.
        loss_3 = 0.
        loss_record = 0.

        total_loss = 0.

        # subgraphs_1 = get_first_adj(dgl_graph, A, subgraph_size_1)
        # subgraphs_2 = get_second_adj(dgl_graph, A, subgraph_size_2)
        #
        # subgraphs_1 = generate_rwr_subgraph(dgl_graph, subgraph_size_1, restart_prob = restart_prob_1)
        # subgraphs_2 = generate_rwr_subgraph(dgl_graph, subgraph_size_2, restart_prob = restart_prob_2)
        subgraphs_1, subgraphs_2 = generate_subgraph(args, dgl_graph, A, subgraph_size_1, subgraph_size_2, epoch)

        for batch_idx in range(batch_num):

            optimiser.zero_grad()

            is_final_batch = (batch_idx == (batch_num - 1))

            # idx = all_idx[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            if not is_final_batch:
                idx = all_idx[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            else:
                idx = all_idx[batch_idx * batch_size:]


            cur_batch_size = len(idx)
            cur_degree = degree[idx]

            lbl = torch.unsqueeze(
                torch.cat((torch.ones(cur_batch_size), torch.zeros(cur_batch_size * args.negsamp_ratio))), 1)

            ba = []
            bf = []
            bf_2 = []
            br = []
            raw = []
            raw_2 = []
            subgraph_idx = []
            subgraph_idx_2 = []

            Z_l = torch.full((cur_batch_size,), 1.)
            added_adj_zero_row = torch.zeros((cur_batch_size, 1, subgraph_size_1))
            added_adj_zero_row_2 = torch.zeros((cur_batch_size, 1, subgraph_size_2))
            added_adj_zero_col = torch.zeros((cur_batch_size, subgraph_size_1 + 1, 1))
            added_adj_zero_col_2 = torch.zeros((cur_batch_size, subgraph_size_2 + 1, 1))
            added_adj_zero_col[:, -1, :] = 1.
            added_adj_zero_col_2[:, -1, :] = 1.
            added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size))

            if torch.cuda.is_available():
                Z_l = Z_l.to(device)
                lbl = lbl.to(device)
                added_adj_zero_row = added_adj_zero_row.to(device)
                added_adj_zero_col = added_adj_zero_col.to(device)
                added_adj_zero_row_2 = added_adj_zero_row_2.to(device)
                added_adj_zero_col_2 = added_adj_zero_col_2.to(device)
                added_feat_zero_row = added_feat_zero_row.to(device)

            for i in idx:
                cur_adj = adj[:, subgraphs_1[i], :][:, :, subgraphs_1[i]]
                cur_adj_r = b_adj[:, subgraphs_2[i], :][:, :, subgraphs_2[i]]
                cur_feat = features[:, subgraphs_1[i], :]
                cur_feat_2 = features[:, subgraphs_2[i], :]
                raw_f = raw_feature[:, subgraphs_1[i], :]
                raw_f_2 = raw_feature[:, subgraphs_2[i], :]
                ba.append(cur_adj)
                br.append(cur_adj_r)
                bf.append(cur_feat)
                bf_2.append(cur_feat_2)
                raw.append(raw_f)
                raw_2.append(raw_f_2)
                subgraph_idx.append(subgraphs_1[i])
                subgraph_idx_2.append(subgraphs_2[i])

            ba = torch.cat(ba)
            br = torch.cat(br)
            ba = torch.cat((ba, added_adj_zero_row), dim=1)
            ba = torch.cat((ba, added_adj_zero_col), dim=2)

            br = torch.cat((br, added_adj_zero_row_2), dim=1)
            br = torch.cat((br, added_adj_zero_col_2), dim=2)


            bf = torch.cat(bf)
            bf = torch.cat((bf[:, :-1, :], added_feat_zero_row, bf[:, -1:, :]), dim=1)
            bf_2 = torch.cat(bf_2)
            bf_2 = torch.cat((bf_2[:, :-1, :], added_feat_zero_row, bf_2[:, -1:, :]), dim=1)

            raw = torch.cat(raw)
            raw = torch.cat((raw[:, :-1, :], added_feat_zero_row, raw[:, -1:, :]), dim=1)
            raw_2 = torch.cat(raw_2)
            raw_2 = torch.cat((raw_2[:, :-1, :], added_feat_zero_row, raw_2[:, -1:, :]), dim=1)

            subgraph_idx = torch.Tensor(subgraph_idx)
            subgraph_idx_2 = torch.Tensor(subgraph_idx_2)
            subgraph_idx = subgraph_idx.int()
            subgraph_idx_2 = subgraph_idx_2.int()
            if torch.cuda.is_available():
                subgraph_idx = subgraph_idx.to(device)
                subgraph_idx_2 = subgraph_idx_2.to(device)

            #/---------------------MODEL-----------------------/#
            node_recons_1, node_recons_2, disc_1, disc_2, inter_loss, _ = \
                model(bf, ba, raw, subgraph_size_1 - 1, bf_2, br, raw_2, subgraph_size_2 - 1)


            loss_recon = 0.5 * (mse_loss(node_recons_1, raw[:, -1, :]) + mse_loss(node_recons_2, raw_2[:, -1, :]))
            intra_loss_1 = b_xent(disc_1, lbl)
            intra_loss_2 = b_xent(disc_2, lbl)
            loss_intra = torch.mean((intra_loss_1 + intra_loss_2) / 2)
            loss_inter = torch.mean(inter_loss)

            # loss_degree_1 = degree_loss_func(degree_logits_1, cur_degree)
            # loss_degree_2 = degree_loss_func(degree_logits_2, cur_degree)
            # loss_degree = 0.5 * (loss_degree_1 + loss_degree_2)

            # loss = beta_recon * loss_recon + alpha_inter * loss_inter
            loss = beta_recon * loss_recon + alpha_inter * loss_inter + loss_intra
                   # + beta_degree * loss_degree
            # print("Reconstruction Loss:{:5}, CoLa Loss:{:5}, OT Loss:{:5}".format(loss_recon, loss_intra, loss_inter))

            loss.backward()
            optimiser.step()

            # if not is_final_batch:
            #     loss_record += loss
            #     loss_1 += loss_recon
            #     loss_2 += loss_inter
            #     loss_3 += loss_intra
            # else:
            #     loss_record_final = loss

            loss = loss.detach().cpu().numpy()
            if not is_final_batch:
                total_loss += loss


        mean_loss = (total_loss * batch_size + loss * cur_batch_size) / nb_nodes


        if mean_loss < best:
            best = mean_loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), 'pkl/best_' + args.dataset + '.pkl')
        else:
            cnt_wait += 1

        # if cnt_wait == args.patience:
        #     print('Early stopping!')
        #     break

        # pbar.set_description("train loss: %.1f" % loss)
        # pbar.set_postfix( {'loss' : '{0:1.5f}\n'.format(mean_loss)} )
        pbar.set_postfix(loss=mean_loss)
        pbar.update(1)




# # Inference phase
# print('Loading {}th epoch'.format(best_t))
path = 'pkl/best_' + args.dataset + '.pkl'
model.load_state_dict(torch.load(path))
multi_round_ano_score = np.zeros((args.auc_test_rounds, nb_nodes))
multi_round_ano_score2 = np.zeros((args.auc_test_rounds, nb_nodes))
multi_round_ano_score3 = np.zeros((args.auc_test_rounds, nb_nodes))
multi_round_ano_score4 = np.zeros((args.auc_test_rounds, nb_nodes))

with tqdm(total=args.auc_test_rounds) as pbar_test:
    pbar_test.set_description('Testing')
    for round in range(args.auc_test_rounds):

        all_idx = list(range(nb_nodes))

        # random.Random(seed).shuffle(all_idx)
        random.shuffle(all_idx)

        subgraphs_1, subgraphs_2 = generate_subgraph(args, dgl_graph, A,subgraph_size_1, subgraph_size_2, round)


        for batch_idx in range(batch_num):

            optimiser.zero_grad()

            is_final_batch = (batch_idx == (batch_num - 1))

            if not is_final_batch:
                idx = all_idx[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            else:
                idx = all_idx[batch_idx * batch_size:]


            cur_batch_size = len(idx)
            cur_degree = degree[idx]

            ba = []
            bf = []
            bf_2 = []
            br = []
            raw = []
            raw_2 = []
            subgraph_idx = []
            subgraph_idx_2 = []
            added_adj_zero_row = torch.zeros((cur_batch_size, 1, subgraph_size_1))
            added_adj_zero_row_2 = torch.zeros((cur_batch_size, 1, subgraph_size_2))
            added_adj_zero_col = torch.zeros((cur_batch_size, subgraph_size_1 + 1, 1))
            added_adj_zero_col_2 = torch.zeros((cur_batch_size, subgraph_size_2 + 1, 1))
            added_adj_zero_col[:, -1, :] = 1.
            added_adj_zero_col_2[:, -1, :] = 1.
            added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size))

            if torch.cuda.is_available():
                added_adj_zero_row = added_adj_zero_row.to(device)
                added_adj_zero_row_2 = added_adj_zero_row_2.to(device)
                added_adj_zero_col = added_adj_zero_col.to(device)
                added_adj_zero_col_2 = added_adj_zero_col_2.to(device)
                added_feat_zero_row = added_feat_zero_row.to(device)

            for i in idx:
                cur_adj = adj[:, subgraphs_1[i], :][:, :, subgraphs_1[i]]
                cur_adj2 = b_adj[:, subgraphs_2[i], :][:, :, subgraphs_2[i]]
                cur_feat = features[:, subgraphs_1[i], :]
                raw_f = raw_feature[:, subgraphs_1[i], :]
                cur_feat_2 = features[:, subgraphs_2[i], :]
                raw_f_2 = raw_feature[:, subgraphs_2[i], :]
                ba.append(cur_adj)
                br.append(cur_adj2)
                bf.append(cur_feat)
                bf_2.append(cur_feat_2)
                raw.append(raw_f)
                raw_2.append(raw_f_2)
                subgraph_idx.append(subgraphs_1[i])
                subgraph_idx_2.append(subgraphs_2[i])

            ba = torch.cat(ba)
            ba = torch.cat((ba, added_adj_zero_row), dim=1)
            ba = torch.cat((ba, added_adj_zero_col), dim=2)
            br = torch.cat(br)
            br = torch.cat((br, added_adj_zero_row_2), dim=1)
            br = torch.cat((br, added_adj_zero_col_2), dim=2)

            bf = torch.cat(bf)
            bf = torch.cat((bf[:, :-1, :], added_feat_zero_row, bf[:, -1:, :]), dim=1)
            bf_2 = torch.cat(bf_2)
            bf_2 = torch.cat((bf_2[:, :-1, :], added_feat_zero_row, bf_2[:, -1:, :]), dim=1)
            raw = torch.cat(raw)
            raw = torch.cat((raw[:, :-1, :], added_feat_zero_row, raw[:, -1:, :]), dim=1)
            raw_2 = torch.cat(raw_2)
            raw_2 = torch.cat((raw_2[:, :-1, :], added_feat_zero_row, raw_2[:, -1:, :]), dim=1)

            subgraph_idx = torch.Tensor(subgraph_idx)
            subgraph_idx_2 = torch.Tensor(subgraph_idx_2)
            subgraph_idx = subgraph_idx.int()
            subgraph_idx_2 = subgraph_idx_2.int()
            if torch.cuda.is_available():
                subgraph_idx = subgraph_idx.to(device)
                subgraph_idx_2 = subgraph_idx_2.to(device)

            # /---------------------MODEL-----------------------/#

            with torch.no_grad():
                node_res_1, node_res_2, logits_1, logits_2, inter_loss, sim_pos = \
                    model(bf, ba, raw, subgraph_size_1 - 1, bf_2, br, raw_2, subgraph_size_2 - 1)


                logits_1 = torch.squeeze(logits_1)
                logits_1 = torch.sigmoid(logits_1)

                logits_2 = torch.squeeze(logits_2)
                logits_2 = torch.sigmoid(logits_2)

            pdist = nn.PairwiseDistance(p=2)
            scaler1 = MinMaxScaler()
            scaler2 = MinMaxScaler()
            scaler3 = MinMaxScaler()
            scaler4 = MinMaxScaler()

            score_co1 = - (logits_1[:cur_batch_size] - logits_1[cur_batch_size:]).cpu().numpy()
            score_co2 = - (logits_2[:cur_batch_size] - logits_2[cur_batch_size:]).cpu().numpy()
            score_co = (score_co1 + score_co2) / 2

            score_re = (pdist(node_res_1, raw[:, -1, :]) + pdist(node_res_2, raw_2[:, -1, :])) / 2
            score_re = score_re.cpu().numpy()

            # score_de1 = (degree_logits_1-cur_degree).pow(2)
            # score_de2 = (degree_logits_2-cur_degree).pow(2)
            # score_de1 = score_de1.cpu().detach()
            # score_de2 = score_de2.cpu().detach()
            # score_de1_norm = score_de1 / (torch.max(score_de1) - torch.min(score_de1))
            # score_de2_norm = score_de2 / (torch.max(score_de2) - torch.min(score_de2))
            # score_de = (score_de1_norm + score_de2_norm) / 2
            # score_degree = score_de.numpy()

            # score_ot = torch.mean(inter_loss, dim=1)
            score_inter = inter_loss.cpu().numpy()
            score_sim = sim_pos.cpu().numpy()


            #nomalize
            ano_score_co = scaler1.fit_transform(score_co.reshape(-1, 1)).reshape(-1)
            ano_score_re = scaler2.fit_transform(score_re.reshape(-1, 1)).reshape(-1)
            ano_score_inter = scaler3.fit_transform(score_inter.reshape(-1, 1)).reshape(-1)
            ano_score_sim = scaler4.fit_transform(score_sim.reshape(-1, 1)).reshape(-1)


            # ano_scores = beta_recon * ano_score_re + alpha_inter * ano_score_ot
            ano_scores = ano_score_co + beta_recon * ano_score_re # anomaly score have no ot
            ano_scores_2 = ano_score_co + beta_recon * ano_score_re + alpha_inter * ano_score_inter # anomaly score have loss(pos+neg)
            ano_scores_3 = ano_score_co + beta_recon * ano_score_re + alpha_inter * ano_score_sim

            # ano_scores_de = ano_score_co + beta_recon * ano_score_re + alpha_inter * ano_score_ot + beta_degree * ano_score_de
            # ano_scores_de2 = ano_score_co + beta_recon * ano_score_re + alpha_inter * ano_score_ot + ano_score_de

            multi_round_ano_score[round, idx] = ano_scores
            multi_round_ano_score2[round, idx] = ano_scores_2
            multi_round_ano_score3[round, idx] = ano_scores_3
            # multi_round_ano_score4[round, idx] = ano_scores_de2


        pbar_test.update(1)
        # auc_cur = roc_auc_score(ano_label, multi_round_ano_score[round, :])
        # AUC_list.append(auc_cur)



ano_score_final = np.mean(multi_round_ano_score, axis=0)
ano_score_final2 = np.mean(multi_round_ano_score2, axis=0)
ano_score_final3 = np.mean(multi_round_ano_score3, axis=0)
# ano_score_final4 = np.mean(multi_round_ano_score4, axis=0)

AUC_ROC = roc_auc_score(ano_label, ano_score_final)
AUC_ROC2 = roc_auc_score(ano_label, ano_score_final2)
AUC_ROC3 = roc_auc_score(ano_label, ano_score_final3)
# AUC_ROC4 = roc_auc_score(ano_label, ano_score_final4)


precision, recall, _ = precision_recall_curve(ano_label, ano_score_final2)
AUPRC = auc(recall, precision)



# print()
print('Dataset: ' + args.dataset)
# print('Mode: ' + args.subgraph_mode)
# print('Seed: ' + str(args.seed))
print("subgraph_size_1:{:1}   subgraph_size_2:{:1}".format(subgraph_size_1, subgraph_size_2))
print("alpha_inter:{:1}   beta_recon:{:1}".format(alpha_inter, beta_recon))
# print('MLP Method on Dataset:'+args.dataset)
# print('Learning rate: ' + str(args.lr))
# print('Top K: ' + str(args.neg_top_k))
# print('Graph neg_sampling round:' + str(graph_negsamp_round))
# print('Batch Size: ' + str(batch_size))
# print('AUC anomaly score without inter loss:  {:.2%}'.format(AUC_ROC))
print('AUC anomaly score with inter loss:     {:.2%}'.format(AUC_ROC2))
# print('AUC anomaly score with similarity:     {:.2%}'.format(AUC_ROC3))
# print('AUC anomaly score with 1 degree loss:  {:.2%}'.format(auc4))
# print('AUC 3:  {:.4f}'.format(auc3))
# print('AUC:  {:.2%}'.format(auc4))
print('AUPRC with inter loss:  {:.2%}'.format(AUPRC))
print('------------------------------')
print()
# AUC_list.append(auc)


# mat_path = './AUC_score'
# os.makedirs(mat_path, exist_ok=True)
# strAUC = str(int(AUC_ROC2*10000))
# sio.savemat(mat_path + '/score_' + args.dataset + '_AUC' + strAUC + '.mat', {'score': ano_score_final2})
