import torch.optim as optim
from model import *
import util
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

# sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj("data/sensor_graph/adj_mx.pkl",args.adjtype)
_,_, normalized = util.load_adj("data/sensor_graph/adj_mx.pkl", "normlap")

class trainer():
    def __init__(self, scaler, in_dim, seq_length, num_nodes, nhid , dropout, lrate, wdecay, device, supports, gcn_bool, addaptadj, aptinit, centrality):
        self.model = gwnet(device, num_nodes, dropout, supports=supports,
         gcn_bool=gcn_bool, addaptadj=addaptadj, aptinit=aptinit,
          in_dim=in_dim, out_dim=seq_length, residual_channels=nhid, 
          dilation_channels=nhid, skip_channels=nhid * 8, end_channels=nhid * 16, centrality=centrality)
        self.model.to(device)
        
        self.loss = util.masked_mae
        self.scaler = scaler
        self.clip = 5
        self.adj_mx = torch.Tensor(normalized[0]).to(device)
        self.device = device

    def train(self, input, real_val, configs):
        self.model.train()
        self.optimizer = optim.Adam(self.model.parameters(), lr=configs["lr"], weight_decay=configs["w_decay"])
        self.optimizer.zero_grad()
        input = nn.functional.pad(input,(1,0,0,0))
        output = self.model(input)
        # output, mid_output = self.model(input)
        output = output.transpose(1,3)
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val,dim=1)
        predict = self.scaler.inverse_transform(output)
        # pred = torch.sigmoid(mid_output.view(mid_output.shape[0] * mid_output.shape[1]))
        # true_label = self.adj_mx.view(mid_output.shape[0] * mid_output.shape[1]).to(self.device)
        # compute_loss = torch.nn.BCELoss()
        # loss_g = compute_loss(pred, true_label)
        loss = self.loss(predict, real, 0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mape,rmse

    def eval(self, input, real_val):
        self.model.eval()
        input = nn.functional.pad(input,(1,0,0,0))
        output = self.model(input)
        # output, mid_output = self.model(input)
        output = output.transpose(1,3)
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val,dim=1)
        predict = self.scaler.inverse_transform(output)
        # pred = torch.sigmoid(mid_output.view(mid_output.shape[0] * mid_output.shape[1]))
        # true_label = self.adj_mx.view(mid_output.shape[0] * mid_output.shape[1]).to(self.device)
        # compute_loss = torch.nn.BCELoss()
        # loss_g = compute_loss(pred, true_label)
        loss = self.loss(predict, real, 0.0)
        tune.report(mean_loss=loss)
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mape,rmse
