import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys

import torch.nn.functional as F
import numpy as np

class SpatialAttention(nn.Module):
    def __init__(self, num_of_timesteps, num_of_features, num_of_vertices):
        super().__init__()
        self.w_1 = nn.Parameter(torch.randn((num_of_timesteps, )))
        self.w_2 = nn.Parameter(torch.randn((num_of_features, num_of_timesteps)))
        self.w_3 = nn.Parameter(torch.randn((num_of_features, )))
        self.b_s = nn.Parameter(torch.randn((1, num_of_vertices, num_of_vertices)))
        self.v_s = nn.Parameter(torch.randn((num_of_vertices, num_of_vertices)))

    def forward(self, x):
        lhs = torch.matmul(torch.matmul(x.view(64, 207, 2, 13),self.w_1), self.w_2)
        # torch.matmul(torch.matmul((x.permute(0, 3, 2, 1).reshape(x.permute(0, 3, 2, 1).size()[2],-1)).T, self.w_2).reshape(64,13,2),
        #              self.U_2)
        # rhs = torch.matmul(self.w_3, x.permute(2,0,3,1))
        rhs = torch.einsum('n,nvlt->vlt', self.w_3 , x.view( 2,64, 13, 207))

        product = torch.matmul(lhs, rhs)
        S =torch.matmul(self.v_s,
                  F.sigmoid(product + self.b_s)
                     .permute(1, 2, 0)).permute(2, 0, 1)
        S = S - torch.max(S, axis=1, keepdims=True)[0]
        exp = torch.exp(S)
        S_normalized = exp / torch.sum(exp, axis=1, keepdims=True)
        return S_normalized

        # lhs = torch.matmul(torch.matmul((x.permute(0, 3, 2, 1).reshape(x.permute(0, 3, 2, 1).size()[2],-1)).T, self.U_1).reshape(64,13,2),
        #              self.U_2)
        # # shape is (N, V, T)
        # # rhs = torch.matmul(self.U_3, x.permute(2, 0, 1, 3))
        # # rhs = (self.U_3 * x.permute(2, 0, 1, 3)).squeeze(0)
        # rhs = torch.einsum('bnlv,v->bnl', (x.permute(2, 0, 1, 3).reshape(207, 64, 13, 2),self.U_3)).contiguous()

        # product = torch.matmul(lhs.reshape(64,13,207), rhs.reshape(64,207,13))

        # E = torch.matmul(self.V_e,
        #            F.sigmoid(product + self.b_e)
        #              .permute(1, 2, 0)).permute(2, 0, 1)

        # # normailzation
        # E = E - torch.max(E, axis=1, keepdims=True)[0]
        # exp = torch.exp(E)
        # E_normalized = exp / torch.sum(exp, axis=1, keepdims=True)
        # return E_normalized

# X = SpatialAttention(3,1,4)

# a = torch.randn(1,4,1,3)
# X(a)

class cheb_conv_with_SAt(nn.Module):
    '''
    K-order chebyshev graph convolution with Spatial Attention scores
    '''
    def __init__(self, num_of_filters, cheb_polynomials, num_of_features,K = 3, **kwargs):
        '''
        Parameters
        ----------
        num_of_filters: int

        num_of_features: int, num of input features

        K: int, up K - 1 order chebyshev polynomials
                will be used in this convolution

        '''
        super(cheb_conv_with_SAt, self).__init__(**kwargs)
        self.K = K
        self.num_of_filters = num_of_filters
        self.cheb_polynomials = cheb_polynomials
        self.Theta = nn.Parameter(torch.randn(self.K, num_of_features, self.num_of_filters))

    def forward(self, x, spatial_attention):
        '''
        Chebyshev graph convolution operation

        Parameters
        ----------
        x: mx.ndarray, graph signal matrix
           shape is (batch_size, N, F, T_{r-1}), F is the num of features

        spatial_attention: mx.ndarray, shape is (batch_size, N, N)
                           spatial attention scores

        Returns
        ----------
        mx.ndarray, shape is (batch_size, N, self.num_of_filters, T_{r-1})

        '''
        (batch_size, num_of_vertices,
         num_of_features, num_of_timesteps) = x.shape

        outputs = []
        for time_step in range(num_of_timesteps):
            # shape is (batch_size, V, F)
            graph_signal = x[:, :, :, time_step]
            output = torch.zeros((batch_size, num_of_vertices,
                                     self.num_of_filters), ctx=x.context)
            for k in range(self.K):

                # shape of T_k is (V, V)
                T_k = self.cheb_polynomials[k]

                # shape of T_k_with_at is (batch_size, V, V)
                T_k_with_at = T_k * spatial_attention

                # shape of theta_k is (F, num_of_filters)
                theta_k = self.Theta[k]

                # shape is (batch_size, V, F)
                rhs = torch.matmul(T_k_with_at.T((0, 2, 1)),
                                   graph_signal)

                output = output + torch.matmul(rhs, theta_k)
            outputs.append(output.expand_dims(-1))
        return F.relu(torch.concat(*outputs, dim=-1))



class Temporal_Attention_layer(nn.Module):
    '''
    compute temporal attention scores
    '''
    def __init__(self,num_of_vertices, num_of_features, num_of_timesteps, **kwargs):
        super(Temporal_Attention_layer, self).__init__(**kwargs)
        self.U_1 = nn.Parameter(torch.randn((num_of_vertices, )))
        self.U_2 = nn.Parameter(torch.randn((num_of_features, num_of_vertices)))
        self.U_3 = nn.Parameter(torch.randn((num_of_features, )))
        self.b_e = nn.Parameter(torch.randn((1, num_of_timesteps, num_of_timesteps)))
        self.V_e = nn.Parameter(torch.randn((num_of_timesteps, num_of_timesteps)))

    def forward(self, x):
        '''
        Parameters
        ----------
        x: mx.ndarray, x^{(r - 1)}_h
                       shape is (batch_size, N, C_{r-1}, T_{r-1})

        Returns
        ----------
        E_normalized: mx.ndarray, S', spatial attention scores
                      shape is (batch_size, T_{r-1}, T_{r-1})

        '''
        _, num_of_vertices, num_of_features, num_of_timesteps = x.shape


        # compute temporal attention scores
        # shape is (N, T, V)
        lhs = torch.matmul(torch.matmul((x.permute(0, 3, 2, 1).reshape(x.permute(0, 3, 2, 1).size()[2],-1)).T, self.U_1).reshape(64,13,2),
                     self.U_2)
        # shape is (N, V, T)
        # rhs = torch.matmul(self.U_3, x.permute(2, 0, 1, 3))
        # rhs = (self.U_3 * x.permute(2, 0, 1, 3)).squeeze(0)
        rhs = torch.einsum('bnlv,v->bnl', (x.permute(2, 0, 1, 3).reshape(207, 64, 13, 2),self.U_3)).contiguous()

        product = torch.matmul(lhs.reshape(64,13,207), rhs.reshape(64,207,13))

        E = torch.matmul(self.V_e,
                   F.sigmoid(product + self.b_e)
                     .permute(1, 2, 0)).permute(2, 0, 1)

        # normailzation
        E = E - torch.max(E, axis=1, keepdims=True)[0]
        exp = torch.exp(E)
        E_normalized = exp / torch.sum(exp, axis=1, keepdims=True)
        return E_normalized

class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)

class gcn(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len=3,order=2):
        super(gcn,self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        out = [x]
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


# Spatial attention

# class gwnet(nn.Module):
#     def __init__(self, device, num_nodes, dropout=0.3, supports=None, gcn_bool=True, addaptadj=True, aptinit=None, in_dim=2,out_dim=12,residual_channels=32,dilation_channels=32,skip_channels=256,end_channels=512,kernel_size=2,blocks=4,layers=2):
#         super(gwnet, self).__init__()
#         self.dropout = dropout
#         self.blocks = blocks
#         self.layers = layers
#         self.gcn_bool = gcn_bool
#         self.addaptadj = addaptadj

#         self.filter_convs = nn.ModuleList()
#         self.gate_convs = nn.ModuleList()
#         self.residual_convs = nn.ModuleList()
#         self.skip_convs = nn.ModuleList()
#         self.bn = nn.ModuleList()
#         self.gconv = nn.ModuleList()
#         self.temporal_attention = Temporal_Attention_layer(num_nodes,2, 13)
#         self.t_h = nn.Parameter(torch.empty((13)))
#         nn.init.uniform_(self.t_h, -(1.0 / np.sqrt(2)), (1.0 / np.sqrt(2)))
#         self.h_x = nn.Parameter(torch.empty(13, 256, 207))
#         nn.init.uniform_(self.h_x,-(1.0 / np.sqrt(2)), (1.0 / np.sqrt(2)))

#         self.start_conv = nn.Conv2d(in_channels=in_dim,
#                                     out_channels=residual_channels,
#                                     kernel_size=(1,1))
#         self.supports = supports

#         receptive_field = 1

#         self.supports_len = 0
#         if supports is not None:
#             self.supports_len += len(supports)

#         if gcn_bool and addaptadj:
#             if aptinit is None:
#                 if supports is None:
#                     self.supports = []
#                 self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
#                 self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
#                 self.supports_len +=1
#             else:
#                 if supports is None:
#                     self.supports = []
#                 m, p, n = torch.svd(aptinit)
#                 initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
#                 initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
#                 self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(device)
#                 self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(device)
#                 self.supports_len += 1




#         for b in range(blocks):
#             additional_scope = kernel_size - 1
#             new_dilation = 1
#             for i in range(layers):
#                 # dilated convolutions
#                 self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
#                                                    out_channels=dilation_channels,
#                                                    kernel_size=(1,kernel_size),dilation=new_dilation))

#                 self.gate_convs.append(nn.Conv1d(in_channels=residual_channels,
#                                                  out_channels=dilation_channels,
#                                                  kernel_size=(1, kernel_size), dilation=new_dilation))

#                 # 1x1 convolution for residual connection
#                 self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
#                                                      out_channels=residual_channels,
#                                                      kernel_size=(1, 1)))

#                 # 1x1 convolution for skip connection
#                 self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
#                                                  out_channels=skip_channels,
#                                                  kernel_size=(1, 1)))
#                 self.bn.append(nn.BatchNorm2d(residual_channels))
#                 new_dilation *=2
#                 receptive_field += additional_scope
#                 additional_scope *= 2
#                 if self.gcn_bool:
#                     self.gconv.append(gcn(dilation_channels,residual_channels,dropout,support_len=self.supports_len))



#         self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
#                                   out_channels=end_channels,
#                                   kernel_size=(1,1),
#                                   bias=True)

#         self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
#                                     out_channels=out_dim,
#                                     kernel_size=(1,1),
#                                     bias=True)

#         self.receptive_field = receptive_field



#     def forward(self, input):
#         tmp = torch.clone(input)
#         in_len = input.size(3)
#         if in_len<self.receptive_field:
#             x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))
#         else:
#             x = input
#         x = self.start_conv(x)
#         skip = 0

#         # calculate the current adaptive adj matrix once per iteration
#         new_supports = None
#         if self.gcn_bool and self.addaptadj and self.supports is not None:
#             adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
#             new_supports = self.supports + [adp]

#         # WaveNet layers
#         for i in range(self.blocks * self.layers):

#             #            |----------------------------------------|     *residual*
#             #            |                                        |
#             #            |    |-- conv -- tanh --|                |
#             # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
#             #                 |-- conv -- sigm --|     |
#             #                                         1x1
#             #                                          |
#             # ---------------------------------------> + ------------->	*skip*

#             #(dilation, init_dilation) = self.dilations[i]

#             #residual = dilation_func(x, dilation, init_dilation, i)
#             residual = x
#             # dilated convolution
#             filter = self.filter_convs[i](residual)
#             filter = torch.tanh(filter)
#             gate = self.gate_convs[i](residual)
#             gate = torch.sigmoid(gate)
#             x = filter * gate

#             # parametrized skip connection

#             s = x
#             s = self.skip_convs[i](s)
#             try:
#                 skip = skip[:, :, :,  -s.size(3):]
#             except:
#                 skip = 0
#             skip = s + skip


#             if self.gcn_bool and self.supports is not None:
#                 if self.addaptadj:
#                     x = self.gconv[i](x, new_supports)
#                 else:
#                     x = self.gconv[i](x,self.supports)
#             else:
#                 x = self.residual_convs[i](x)

#             x = x + residual[:, :, :, -x.size(3):]


#             x = self.bn[i](x)

#         x = F.relu(skip)
#         x = x + torch.einsum('nc,cva->nva', (self.temporal_attention(tmp) @ self.t_h, self.h_x)).contiguous().unsqueeze(-1)
#         x = F.relu(self.end_conv_1(x))
#         x = self.end_conv_2(x)
#         return x


# Spatial attention

# class gwnet(nn.Module):
#     def __init__(self, device, num_nodes, dropout=0.3, supports=None, gcn_bool=True, addaptadj=True, aptinit=None, in_dim=2,out_dim=12,residual_channels=32,dilation_channels=32,skip_channels=256,end_channels=512,kernel_size=2,blocks=4,layers=2):
#         super(gwnet, self).__init__()
#         self.dropout = dropout
#         self.blocks = blocks
#         self.layers = layers
#         self.gcn_bool = gcn_bool
#         self.addaptadj = addaptadj

#         self.filter_convs = nn.ModuleList()
#         self.gate_convs = nn.ModuleList()
#         self.residual_convs = nn.ModuleList()
#         self.skip_convs = nn.ModuleList()
#         self.bn = nn.ModuleList()
#         self.gconv = nn.ModuleList()
#         self.spatial_attention = SpatialAttention(13,2, num_nodes)
#         self.t_h = nn.Parameter(torch.empty((207)))
#         nn.init.uniform_(self.t_h, -(1.0 / np.sqrt(2)), (1.0 / np.sqrt(2)))
#         self.h_x = nn.Parameter(torch.empty(207, 256, 207))
#         nn.init.uniform_(self.h_x,-(1.0 / np.sqrt(2)), (1.0 / np.sqrt(2)))

#         self.start_conv = nn.Conv2d(in_channels=in_dim,
#                                     out_channels=residual_channels,
#                                     kernel_size=(1,1))
#         self.supports = supports

#         receptive_field = 1

#         self.supports_len = 0
#         if supports is not None:
#             self.supports_len += len(supports)

#         if gcn_bool and addaptadj:
#             if aptinit is None:
#                 if supports is None:
#                     self.supports = []
#                 self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
#                 self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
#                 self.supports_len +=1
#             else:
#                 if supports is None:
#                     self.supports = []
#                 m, p, n = torch.svd(aptinit)
#                 initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
#                 initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
#                 self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(device)
#                 self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(device)
#                 self.supports_len += 1




#         for b in range(blocks):
#             additional_scope = kernel_size - 1
#             new_dilation = 1
#             for i in range(layers):
#                 # dilated convolutions
#                 self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
#                                                    out_channels=dilation_channels,
#                                                    kernel_size=(1,kernel_size),dilation=new_dilation))

#                 self.gate_convs.append(nn.Conv1d(in_channels=residual_channels,
#                                                  out_channels=dilation_channels,
#                                                  kernel_size=(1, kernel_size), dilation=new_dilation))

#                 # 1x1 convolution for residual connection
#                 self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
#                                                      out_channels=residual_channels,
#                                                      kernel_size=(1, 1)))

#                 # 1x1 convolution for skip connection
#                 self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
#                                                  out_channels=skip_channels,
#                                                  kernel_size=(1, 1)))
#                 self.bn.append(nn.BatchNorm2d(residual_channels))
#                 new_dilation *=2
#                 receptive_field += additional_scope
#                 additional_scope *= 2
#                 if self.gcn_bool:
#                     self.gconv.append(gcn(dilation_channels,residual_channels,dropout,support_len=self.supports_len))



#         self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
#                                   out_channels=end_channels,
#                                   kernel_size=(1,1),
#                                   bias=True)

#         self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
#                                     out_channels=out_dim,
#                                     kernel_size=(1,1),
#                                     bias=True)

#         self.receptive_field = receptive_field



#     def forward(self, input):
#         tmp = torch.clone(input)
#         in_len = input.size(3)
#         if in_len<self.receptive_field:
#             x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))
#         else:
#             x = input
#         x = self.start_conv(x)
#         skip = 0

#         # calculate the current adaptive adj matrix once per iteration
#         new_supports = None
#         if self.gcn_bool and self.addaptadj and self.supports is not None:
#             adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
#             new_supports = self.supports + [adp]

#         # WaveNet layers
#         for i in range(self.blocks * self.layers):

#             #            |----------------------------------------|     *residual*
#             #            |                                        |
#             #            |    |-- conv -- tanh --|                |
#             # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
#             #                 |-- conv -- sigm --|     |
#             #                                         1x1
#             #                                          |
#             # ---------------------------------------> + ------------->	*skip*

#             #(dilation, init_dilation) = self.dilations[i]

#             #residual = dilation_func(x, dilation, init_dilation, i)
#             residual = x
#             # dilated convolution
#             filter = self.filter_convs[i](residual)
#             filter = torch.tanh(filter)
#             gate = self.gate_convs[i](residual)
#             gate = torch.sigmoid(gate)
#             x = filter * gate

#             # parametrized skip connection

#             s = x
#             s = self.skip_convs[i](s)
#             try:
#                 skip = skip[:, :, :,  -s.size(3):]
#             except:
#                 skip = 0
#             skip = s + skip


#             if self.gcn_bool and self.supports is not None:
#                 if self.addaptadj:
#                     x = self.gconv[i](x, new_supports)
#                 else:
#                     x = self.gconv[i](x,self.supports)
#             else:
#                 x = self.residual_convs[i](x)

#             x = x + residual[:, :, :, -x.size(3):]


#             x = self.bn[i](x)

#         x = F.relu(skip)
#         x = x + torch.einsum('nc,cva->nva', (self.spatial_attention(tmp) @ self.t_h, self.h_x)).contiguous().unsqueeze(-1)
#         x = F.relu(self.end_conv_1(x))
#         x = self.end_conv_2(x)
#         return x




# class gwnet(nn.Module):
#     def __init__(self, device, num_nodes, dropout=0.3, supports=None, gcn_bool=True, addaptadj=True, aptinit=None, in_dim=2,out_dim=12,residual_channels=32,dilation_channels=32,skip_channels=256,end_channels=512,kernel_size=2,blocks=4,layers=2):
#         super(gwnet, self).__init__()
#         self.dropout = dropout
#         self.blocks = blocks
#         self.layers = layers
#         self.gcn_bool = gcn_bool
#         self.addaptadj = addaptadj

#         self.filter_convs = nn.ModuleList()
#         self.gate_convs = nn.ModuleList()
#         self.residual_convs = nn.ModuleList()
#         self.skip_convs = nn.ModuleList()
#         self.bn = nn.ModuleList()
#         self.gconv = nn.ModuleList()

#         self.spatial_attention = SpatialAttention(13,2, num_nodes)
#         self.temporal_attention = Temporal_Attention_layer(num_nodes,2, 13)

#         self.t_h = nn.Parameter(torch.empty((13)))
#         nn.init.uniform_(self.t_h, -(1.0 / np.sqrt(2)), (1.0 / np.sqrt(2)))
#         self.h_x = nn.Parameter(torch.empty(13, 256, 207))
#         nn.init.uniform_(self.h_x,-(1.0 / np.sqrt(2)), (1.0 / np.sqrt(2)))

#         self.t_h_sp = nn.Parameter(torch.empty((207)))
#         nn.init.uniform_(self.t_h_sp, -(1.0 / np.sqrt(2)), (1.0 / np.sqrt(2)))
#         self.h_x_sp = nn.Parameter(torch.empty(207, 256, 207))
#         nn.init.uniform_(self.h_x_sp,-(1.0 / np.sqrt(2)), (1.0 / np.sqrt(2)))

#         self.start_conv = nn.Conv2d(in_channels=in_dim,
#                                     out_channels=residual_channels,
#                                     kernel_size=(1,1))
#         self.supports = supports

#         receptive_field = 1

#         self.supports_len = 0
#         if supports is not None:
#             self.supports_len += len(supports)

#         if gcn_bool and addaptadj:
#             if aptinit is None:
#                 if supports is None:
#                     self.supports = []
#                 self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
#                 self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
#                 self.supports_len +=1
#             else:
#                 if supports is None:
#                     self.supports = []
#                 m, p, n = torch.svd(aptinit)
#                 initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
#                 initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
#                 self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(device)
#                 self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(device)
#                 self.supports_len += 1




#         for b in range(blocks):
#             additional_scope = kernel_size - 1
#             new_dilation = 1
#             for i in range(layers):
#                 # dilated convolutions
#                 self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
#                                                    out_channels=dilation_channels,
#                                                    kernel_size=(1,kernel_size),dilation=new_dilation))

#                 self.gate_convs.append(nn.Conv1d(in_channels=residual_channels,
#                                                  out_channels=dilation_channels,
#                                                  kernel_size=(1, kernel_size), dilation=new_dilation))

#                 # 1x1 convolution for residual connection
#                 self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
#                                                      out_channels=residual_channels,
#                                                      kernel_size=(1, 1)))

#                 # 1x1 convolution for skip connection
#                 self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
#                                                  out_channels=skip_channels,
#                                                  kernel_size=(1, 1)))
#                 self.bn.append(nn.BatchNorm2d(residual_channels))
#                 new_dilation *=2
#                 receptive_field += additional_scope
#                 additional_scope *= 2
#                 if self.gcn_bool:
#                     self.gconv.append(gcn(dilation_channels,residual_channels,dropout,support_len=self.supports_len))



#         self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
#                                   out_channels=end_channels,
#                                   kernel_size=(1,1),
#                                   bias=True)

#         self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
#                                     out_channels=out_dim,
#                                     kernel_size=(1,1),
#                                     bias=True)

#         self.receptive_field = receptive_field



#     def forward(self, input):
#         tmp = torch.clone(input)
#         in_len = input.size(3)
#         if in_len<self.receptive_field:
#             x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))
#         else:
#             x = input
#         x = self.start_conv(x)
#         skip = 0

#         # calculate the current adaptive adj matrix once per iteration
#         new_supports = None
#         if self.gcn_bool and self.addaptadj and self.supports is not None:
#             adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
#             new_supports = self.supports + [adp]

#         # WaveNet layers
#         for i in range(self.blocks * self.layers):

#             #            |----------------------------------------|     *residual*
#             #            |                                        |
#             #            |    |-- conv -- tanh --|                |
#             # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
#             #                 |-- conv -- sigm --|     |
#             #                                         1x1
#             #                                          |
#             # ---------------------------------------> + ------------->	*skip*

#             #(dilation, init_dilation) = self.dilations[i]

#             #residual = dilation_func(x, dilation, init_dilation, i)
#             residual = x
#             # dilated convolution
#             filter = self.filter_convs[i](residual)
#             filter = torch.tanh(filter)
#             gate = self.gate_convs[i](residual)
#             gate = torch.sigmoid(gate)
#             x = filter * gate

#             # parametrized skip connection

#             s = x
#             s = self.skip_convs[i](s)
#             try:
#                 skip = skip[:, :, :,  -s.size(3):]
#             except:
#                 skip = 0
#             skip = s + skip


#             if self.gcn_bool and self.supports is not None:
#                 if self.addaptadj:
#                     x = self.gconv[i](x, new_supports)
#                 else:
#                     x = self.gconv[i](x,self.supports)
#             else:
#                 x = self.residual_convs[i](x)

#             x = x + residual[:, :, :, -x.size(3):]


#             x = self.bn[i](x)

#         x = F.relu(skip)
#         x = x + torch.einsum('nc,cva->nva', (self.spatial_attention(tmp) @ self.t_h_sp , self.h_x_sp)).contiguous().unsqueeze(-1) + \
#             torch.einsum('nc,cva->nva', (self.temporal_attention(tmp) @ self.t_h, self.h_x)).contiguous().unsqueeze(-1)
#         x = F.relu(self.end_conv_1(x))
#         x = self.end_conv_2(x)
#         return x

# class gwnet(nn.Module):
#     def __init__(self, device, num_nodes, dropout=0.3, supports=None, gcn_bool=True, addaptadj=True, aptinit=None, in_dim=2,out_dim=12,residual_channels=32,dilation_channels=32,skip_channels=256,end_channels=512,kernel_size=2,blocks=4,layers=2):
#         super(gwnet, self).__init__()
#         self.dropout = dropout
#         self.blocks = blocks
#         self.layers = layers
#         self.gcn_bool = gcn_bool
#         self.addaptadj = addaptadj

#         self.filter_convs = nn.ModuleList()
#         self.gate_convs = nn.ModuleList()
#         self.residual_convs = nn.ModuleList()
#         self.skip_convs = nn.ModuleList()
#         self.bn = nn.ModuleList()
#         self.gconv = nn.ModuleList()

#         self.spatial_attention = SpatialAttention(13,2, num_nodes)
#         self.temporal_attention = Temporal_Attention_layer(num_nodes,2, 13)

#         self.t_h = nn.Parameter(torch.empty((13)))
#         nn.init.uniform_(self.t_h, -(1.0 / np.sqrt(2)), (1.0 / np.sqrt(2)))
#         self.h_x = nn.Parameter(torch.empty(13, 256, 207))
#         nn.init.uniform_(self.h_x,-(1.0 / np.sqrt(2)), (1.0 / np.sqrt(2)))

#         self.t_h_sp = nn.Parameter(torch.empty((207)))
#         nn.init.uniform_(self.t_h_sp, -(1.0 / np.sqrt(2)), (1.0 / np.sqrt(2)))
#         self.h_x_sp = nn.Parameter(torch.empty(207, 256, 207))
#         nn.init.uniform_(self.h_x_sp,-(1.0 / np.sqrt(2)), (1.0 / np.sqrt(2)))

#         self.t_we = nn.Parameter(torch.empty((1)))

#         self.s_we = nn.Parameter(torch.empty((1)))
#         nn.init.uniform_(self.t_we,0,1)
#         nn.init.uniform_(self.s_we,0,1)
#         self.start_conv = nn.Conv2d(in_channels=in_dim,
#                                     out_channels=residual_channels,
#                                     kernel_size=(1,1))
#         self.supports = supports

#         receptive_field = 1

#         self.supports_len = 0
#         if supports is not None:
#             self.supports_len += len(supports)

#         if gcn_bool and addaptadj:
#             if aptinit is None:
#                 if supports is None:
#                     self.supports = []
#                 self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
#                 self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
#                 self.supports_len +=1
#             else:
#                 if supports is None:
#                     self.supports = []
#                 m, p, n = torch.svd(aptinit)
#                 initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
#                 initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
#                 self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(device)
#                 self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(device)
#                 self.supports_len += 1




#         for b in range(blocks):
#             additional_scope = kernel_size - 1
#             new_dilation = 1
#             for i in range(layers):
#                 # dilated convolutions
#                 self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
#                                                    out_channels=dilation_channels,
#                                                    kernel_size=(1,kernel_size),dilation=new_dilation))

#                 self.gate_convs.append(nn.Conv1d(in_channels=residual_channels,
#                                                  out_channels=dilation_channels,
#                                                  kernel_size=(1, kernel_size), dilation=new_dilation))

#                 # 1x1 convolution for residual connection
#                 self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
#                                                      out_channels=residual_channels,
#                                                      kernel_size=(1, 1)))

#                 # 1x1 convolution for skip connection
#                 self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
#                                                  out_channels=skip_channels,
#                                                  kernel_size=(1, 1)))
#                 self.bn.append(nn.BatchNorm2d(residual_channels))
#                 new_dilation *=2
#                 receptive_field += additional_scope
#                 additional_scope *= 2
#                 if self.gcn_bool:
#                     self.gconv.append(gcn(dilation_channels,residual_channels,dropout,support_len=self.supports_len))



#         self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
#                                   out_channels=end_channels,
#                                   kernel_size=(1,1),
#                                   bias=True)

#         self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
#                                     out_channels=out_dim,
#                                     kernel_size=(1,1),
#                                     bias=True)

#         self.receptive_field = receptive_field



#     def forward(self, input):
#         tmp = torch.clone(input)
#         in_len = input.size(3)
#         if in_len<self.receptive_field:
#             x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))
#         else:
#             x = input
#         x = self.start_conv(x)
#         skip = 0

#         # calculate the current adaptive adj matrix once per iteration
#         new_supports = None
#         if self.gcn_bool and self.addaptadj and self.supports is not None:
#             adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
#             new_supports = self.supports + [adp]

#         # WaveNet layers
#         for i in range(self.blocks * self.layers):

#             #            |----------------------------------------|     *residual*
#             #            |                                        |
#             #            |    |-- conv -- tanh --|                |
#             # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
#             #                 |-- conv -- sigm --|     |
#             #                                         1x1
#             #                                          |
#             # ---------------------------------------> + ------------->	*skip*

#             #(dilation, init_dilation) = self.dilations[i]

#             #residual = dilation_func(x, dilation, init_dilation, i)
#             residual = x
#             # dilated convolution
#             filter = self.filter_convs[i](residual)
#             filter = torch.tanh(filter)
#             gate = self.gate_convs[i](residual)
#             gate = torch.sigmoid(gate)
#             x = filter * gate

#             # parametrized skip connection

#             s = x
#             s = self.skip_convs[i](s)
#             try:
#                 skip = skip[:, :, :,  -s.size(3):]
#             except:
#                 skip = 0
#             skip = s + skip


#             if self.gcn_bool and self.supports is not None:
#                 if self.addaptadj:
#                     x = self.gconv[i](x, new_supports)
#                 else:
#                     x = self.gconv[i](x,self.supports)
#             else:
#                 x = self.residual_convs[i](x)

#             x = x + residual[:, :, :, -x.size(3):]


#             x = self.bn[i](x)

#         x = F.relu(skip)
#         x = x + self.s_we * torch.einsum('nc,cva->nva', (self.spatial_attention(tmp) @ self.t_h_sp , self.h_x_sp)).contiguous().unsqueeze(-1) + \
#             self.t_we * torch.einsum('nc,cva->nva', (self.temporal_attention(tmp) @ self.t_h, self.h_x)).contiguous().unsqueeze(-1)
#         x = F.relu(self.end_conv_1(x))
#         x = self.end_conv_2(x)
#         return x



# Seq S->T
# class gwnet(nn.Module):
#     def __init__(self, device, num_nodes, dropout=0.3, supports=None, gcn_bool=True, addaptadj=True, aptinit=None, in_dim=2,out_dim=12,residual_channels=32,dilation_channels=32,skip_channels=256,end_channels=512,kernel_size=2,blocks=4,layers=2):
#         super(gwnet, self).__init__()
#         self.dropout = dropout
#         self.blocks = blocks
#         self.layers = layers
#         self.gcn_bool = gcn_bool
#         self.addaptadj = addaptadj

#         self.filter_convs = nn.ModuleList()
#         self.gate_convs = nn.ModuleList()
#         self.residual_convs = nn.ModuleList()
#         self.skip_convs = nn.ModuleList()
#         self.bn = nn.ModuleList()
#         self.gconv = nn.ModuleList()

#         self.spatial_attention = SpatialAttention(13,2, num_nodes)
#         self.temporal_attention = Temporal_Attention_layer(num_nodes,2, 13)

#         self.t_h = nn.Parameter(torch.empty((13)))
#         nn.init.uniform_(self.t_h, -(1.0 / np.sqrt(2)), (1.0 / np.sqrt(2)))
#         self.h_x = nn.Parameter(torch.empty(13, 256, 207))
#         nn.init.uniform_(self.h_x,-(1.0 / np.sqrt(2)), (1.0 / np.sqrt(2)))

#         self.t_1 = nn.Parameter(torch.empty(64,207,13))
#         nn.init.uniform_(self.t_1,-(1.0 / np.sqrt(2)), (1.0 / np.sqrt(2)))
#         self.t_2 = nn.Parameter(torch.empty(64,1,1,2))
#         nn.init.uniform_(self.t_1,-(1.0 / np.sqrt(2)), (1.0 / np.sqrt(2)))


#         self.t_h_sp = nn.Parameter(torch.empty((207)))
#         nn.init.uniform_(self.t_h_sp, -(1.0 / np.sqrt(2)), (1.0 / np.sqrt(2)))
#         self.h_x_sp = nn.Parameter(torch.empty(207, 256, 207))
#         nn.init.uniform_(self.h_x_sp,-(1.0 / np.sqrt(2)), (1.0 / np.sqrt(2)))

#         self.t_we = nn.Parameter(torch.empty((1)))

#         self.s_we = nn.Parameter(torch.empty((1)))
#         nn.init.uniform_(self.t_we,0,1)
#         nn.init.uniform_(self.s_we,0,1)
#         self.start_conv = nn.Conv2d(in_channels=in_dim,
#                                     out_channels=residual_channels,
#                                     kernel_size=(1,1))
#         self.supports = supports

#         receptive_field = 1

#         self.supports_len = 0
#         if supports is not None:
#             self.supports_len += len(supports)

#         if gcn_bool and addaptadj:
#             if aptinit is None:
#                 if supports is None:
#                     self.supports = []
#                 self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
#                 self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
#                 self.supports_len +=1
#             else:
#                 if supports is None:
#                     self.supports = []
#                 m, p, n = torch.svd(aptinit)
#                 initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
#                 initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
#                 self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(device)
#                 self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(device)
#                 self.supports_len += 1




#         for b in range(blocks):
#             additional_scope = kernel_size - 1
#             new_dilation = 1
#             for i in range(layers):
#                 # dilated convolutions
#                 self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
#                                                    out_channels=dilation_channels,
#                                                    kernel_size=(1,kernel_size),dilation=new_dilation))

#                 self.gate_convs.append(nn.Conv1d(in_channels=residual_channels,
#                                                  out_channels=dilation_channels,
#                                                  kernel_size=(1, kernel_size), dilation=new_dilation))

#                 # 1x1 convolution for residual connection
#                 self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
#                                                      out_channels=residual_channels,
#                                                      kernel_size=(1, 1)))

#                 # 1x1 convolution for skip connection
#                 self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
#                                                  out_channels=skip_channels,
#                                                  kernel_size=(1, 1)))
#                 self.bn.append(nn.BatchNorm2d(residual_channels))
#                 new_dilation *=2
#                 receptive_field += additional_scope
#                 additional_scope *= 2
#                 if self.gcn_bool:
#                     self.gconv.append(gcn(dilation_channels,residual_channels,dropout,support_len=self.supports_len))



#         self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
#                                   out_channels=end_channels,
#                                   kernel_size=(1,1),
#                                   bias=True)

#         self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
#                                     out_channels=out_dim,
#                                     kernel_size=(1,1),
#                                     bias=True)

#         self.receptive_field = receptive_field



#     def forward(self, input):
#         tmp = torch.clone(input)
#         in_len = input.size(3)
#         if in_len<self.receptive_field:
#             x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))
#         else:
#             x = input
#         x = self.start_conv(x)
#         skip = 0

#         # calculate the current adaptive adj matrix once per iteration
#         new_supports = None
#         if self.gcn_bool and self.addaptadj and self.supports is not None:
#             adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
#             new_supports = self.supports + [adp]

#         # WaveNet layers
#         for i in range(self.blocks * self.layers):

#             #            |----------------------------------------|     *residual*
#             #            |                                        |
#             #            |    |-- conv -- tanh --|                |
#             # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
#             #                 |-- conv -- sigm --|     |
#             #                                         1x1
#             #                                          |
#             # ---------------------------------------> + ------------->	*skip*

#             #(dilation, init_dilation) = self.dilations[i]

#             #residual = dilation_func(x, dilation, init_dilation, i)
#             residual = x
#             # dilated convolution
#             filter = self.filter_convs[i](residual)
#             filter = torch.tanh(filter)
#             gate = self.gate_convs[i](residual)
#             gate = torch.sigmoid(gate)
#             x = filter * gate

#             # parametrized skip connection

#             s = x
#             s = self.skip_convs[i](s)
#             try:
#                 skip = skip[:, :, :,  -s.size(3):]
#             except:
#                 skip = 0
#             skip = s + skip


#             if self.gcn_bool and self.supports is not None:
#                 if self.addaptadj:
#                     x = self.gconv[i](x, new_supports)
#                 else:
#                     x = self.gconv[i](x,self.supports)
#             else:
#                 x = self.residual_convs[i](x)

#             x = x + residual[:, :, :, -x.size(3):]


#             x = self.bn[i](x)

#         x = F.relu(skip)
#         ttt = torch.matmul(self.t_1,self.temporal_attention(tmp))
#         # ttt = torch.matmul(torch.randn(64,207,13).to("cuda:0"),self.temporal_attention(tmp))

#         input_of_temporal_att = torch.matmul(ttt.unsqueeze(-1) ,self.t_2).view(64,2,207,13)
#         # input_of_sp_att = torch.matmul(ttt.unsqueeze(-1) ,torch.randn(64,1,1,2).to("cuda:0")).view(64,2,207,13)

#         x = x + self.s_we * torch.einsum('nc,cva->nva', (self.spatial_attention(input_of_temporal_att) @ self.t_h_sp , self.h_x_sp)).contiguous().unsqueeze(-1)
#         x = F.relu(self.end_conv_1(x))
#         x = self.end_conv_2(x)
#         return x


# class gwnet(nn.Module):
#     def __init__(self, device, num_nodes, dropout=0.3, supports=None, gcn_bool=True, addaptadj=True, aptinit=None, in_dim=2,out_dim=12,residual_channels=32,dilation_channels=32,skip_channels=256,end_channels=512,kernel_size=2,blocks=4,layers=2):
#         super(gwnet, self).__init__()
#         self.dropout = dropout
#         self.blocks = blocks
#         self.layers = layers
#         self.gcn_bool = gcn_bool
#         self.addaptadj = addaptadj

#         self.filter_convs = nn.ModuleList()
#         self.gate_convs = nn.ModuleList()
#         self.residual_convs = nn.ModuleList()
#         self.skip_convs = nn.ModuleList()
#         self.bn = nn.ModuleList()
#         self.gconv = nn.ModuleList()

#         self.spatial_attention = SpatialAttention(13,2, num_nodes)
#         self.temporal_attention = Temporal_Attention_layer(num_nodes,2, 13)

#         self.t_h = nn.Parameter(torch.empty((13)))
#         nn.init.uniform_(self.t_h, -(1.0 / np.sqrt(2)), (1.0 / np.sqrt(2)))
#         self.h_x = nn.Parameter(torch.empty(13, 256, 207))
#         nn.init.uniform_(self.h_x,-(1.0 / np.sqrt(2)), (1.0 / np.sqrt(2)))

#         self.t_1 = nn.Parameter(torch.empty(64,207,13))
#         nn.init.uniform_(self.t_1,-(1.0 / np.sqrt(2)), (1.0 / np.sqrt(2)))
#         self.t_2 = nn.Parameter(torch.empty(64,1,1,2))
#         nn.init.uniform_(self.t_1,-(1.0 / np.sqrt(2)), (1.0 / np.sqrt(2)))


#         # self.t_h_sp = nn.Parameter(torch.empty((207)))
#         # nn.init.uniform_(self.t_h_sp, -(1.0 / np.sqrt(2)), (1.0 / np.sqrt(2)))
#         self.h_x_sp = nn.Parameter(torch.empty(64,13,207))
#         nn.init.uniform_(self.h_x_sp,-(1.0 / np.sqrt(2)), (1.0 / np.sqrt(2)))

#         self.t_we = nn.Parameter(torch.empty((1)))

#         self.s_we = nn.Parameter(torch.empty((1)))
#         nn.init.uniform_(self.t_we,0,1)
#         nn.init.uniform_(self.s_we,0,1)
#         self.start_conv = nn.Conv2d(in_channels=in_dim,
#                                     out_channels=residual_channels,
#                                     kernel_size=(1,1))
#         self.supports = supports

#         receptive_field = 1

#         self.supports_len = 0
#         if supports is not None:
#             self.supports_len += len(supports)

#         if gcn_bool and addaptadj:
#             if aptinit is None:
#                 if supports is None:
#                     self.supports = []
#                 self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
#                 self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
#                 self.supports_len +=1
#             else:
#                 if supports is None:
#                     self.supports = []
#                 m, p, n = torch.svd(aptinit)
#                 initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
#                 initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
#                 self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(device)
#                 self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(device)
#                 self.supports_len += 1




#         for b in range(blocks):
#             additional_scope = kernel_size - 1
#             new_dilation = 1
#             for i in range(layers):
#                 # dilated convolutions
#                 self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
#                                                    out_channels=dilation_channels,
#                                                    kernel_size=(1,kernel_size),dilation=new_dilation))

#                 self.gate_convs.append(nn.Conv1d(in_channels=residual_channels,
#                                                  out_channels=dilation_channels,
#                                                  kernel_size=(1, kernel_size), dilation=new_dilation))

#                 # 1x1 convolution for residual connection
#                 self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
#                                                      out_channels=residual_channels,
#                                                      kernel_size=(1, 1)))

#                 # 1x1 convolution for skip connection
#                 self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
#                                                  out_channels=skip_channels,
#                                                  kernel_size=(1, 1)))
#                 self.bn.append(nn.BatchNorm2d(residual_channels))
#                 new_dilation *=2
#                 receptive_field += additional_scope
#                 additional_scope *= 2
#                 if self.gcn_bool:
#                     self.gconv.append(gcn(dilation_channels,residual_channels,dropout,support_len=self.supports_len))



#         self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
#                                   out_channels=end_channels,
#                                   kernel_size=(1,1),
#                                   bias=True)

#         self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
#                                     out_channels=out_dim,
#                                     kernel_size=(1,1),
#                                     bias=True)
        
#         self.linear = nn.Linear(220,256)

#         self.receptive_field = receptive_field



#     def forward(self, input):
#         tmp = torch.clone(input)
#         in_len = input.size(3)
#         if in_len<self.receptive_field:
#             x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))
#         else:
#             x = input
#         x = self.start_conv(x)
#         skip = 0

#         # calculate the current adaptive adj matrix once per iteration
#         new_supports = None
#         if self.gcn_bool and self.addaptadj and self.supports is not None:
#             adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
#             new_supports = self.supports + [adp]

#         # WaveNet layers
#         for i in range(self.blocks * self.layers):

#             #            |----------------------------------------|     *residual*
#             #            |                                        |
#             #            |    |-- conv -- tanh --|                |
#             # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
#             #                 |-- conv -- sigm --|     |
#             #                                         1x1
#             #                                          |
#             # ---------------------------------------> + ------------->	*skip*

#             #(dilation, init_dilation) = self.dilations[i]

#             #residual = dilation_func(x, dilation, init_dilation, i)
#             residual = x
#             # dilated convolution
#             filter = self.filter_convs[i](residual)
#             filter = torch.tanh(filter)
#             gate = self.gate_convs[i](residual)
#             gate = torch.sigmoid(gate)
#             x = filter * gate

#             # parametrized skip connection

#             s = x
#             s = self.skip_convs[i](s)
#             try:
#                 skip = skip[:, :, :,  -s.size(3):]
#             except:
#                 skip = 0
#             skip = s + skip


#             if self.gcn_bool and self.supports is not None:
#                 if self.addaptadj:
#                     x = self.gconv[i](x, new_supports)
#                 else:
#                     x = self.gconv[i](x,self.supports)
#             else:
#                 x = self.residual_convs[i](x)

#             x = x + residual[:, :, :, -x.size(3):]


#             x = self.bn[i](x)

#         x = F.relu(skip)
#         # ttt = torch.matmul(self.t_1,self.temporal_attention(tmp))
#         ttt = self.temporal_attention(tmp).matmul(self.h_x_sp)
#         # ttt = torch.matmul(torch.randn(64,207,13).to("cuda:0"),self.temporal_attention(tmp))
#         concat = torch.cat((self.spatial_attention(tmp),ttt),1).view(64,207,220)
#         linear_out = self.linear(concat).view(64,256,207).unsqueeze(-1)
#         # input_of_temporal_att = torch.matmul(ttt.unsqueeze(-1) ,self.t_2).view(64,2,207,13)
#         # input_of_sp_att = torch.matmul(ttt.unsqueeze(-1) ,torch.randn(64,1,1,2).to("cuda:0")).view(64,2,207,13)
#         # import pdb;pdb.set_trace()
#         # x shape = torch.Size([64, 256, 207, 1])
#         # x = x + self.s_we * torch.einsum('nc,cva->nva', ( @ self.t_h_sp , self.h_x_sp)).contiguous().unsqueeze(-1)
#         x = x + self.t_we * linear_out
#         x = F.relu(self.end_conv_1(x))
#         x = self.end_conv_2(x)
#         return x




class gwnet(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.3, supports=None, gcn_bool=True, addaptadj=True, aptinit=None, in_dim=2,out_dim=12,residual_channels=32,dilation_channels=32,skip_channels=256,end_channels=512,kernel_size=2,blocks=4,layers=2, sp=None, centrality=None):
        super(gwnet, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.spatial_attention = SpatialAttention(13,2, num_nodes)
        self.temporal_attention = Temporal_Attention_layer(num_nodes,2, 13)

        self.t_h = nn.Parameter(torch.empty((13)))
        nn.init.uniform_(self.t_h, -(1.0 / np.sqrt(2)), (1.0 / np.sqrt(2)))
        self.h_x = nn.Parameter(torch.empty(13, 256, 207))
        nn.init.uniform_(self.h_x,-(1.0 / np.sqrt(2)), (1.0 / np.sqrt(2)))

        self.t_1 = nn.Parameter(torch.empty(64,207,13))
        nn.init.uniform_(self.t_1,-(1.0 / np.sqrt(2)), (1.0 / np.sqrt(2)))
        self.t_2 = nn.Parameter(torch.empty(64,1,1,2))
        nn.init.uniform_(self.t_1,-(1.0 / np.sqrt(2)), (1.0 / np.sqrt(2)))

        # self.sa = nn.MultiheadAttention(32,8)
        # self.t_h_sp = nn.Parameter(torch.empty((207)))
        # nn.init.uniform_(self.t_h_sp, -(1.0 / np.sqrt(2)), (1.0 / np.sqrt(2)))
        self.h_x_sp = nn.Parameter(torch.empty(64,13,207))
        nn.init.uniform_(self.h_x_sp,-(1.0 / np.sqrt(2)), (1.0 / np.sqrt(2)))

        self.t_we = nn.Parameter(torch.empty((1)))

        self.s_we = nn.Parameter(torch.empty((1)))
        nn.init.uniform_(self.t_we,0,1)
        nn.init.uniform_(self.s_we,0,1)
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))
        self.supports = supports

        receptive_field = 1

        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        if gcn_bool and addaptadj:
            if aptinit is None:
                if supports is None:
                    self.supports = []
                self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
                self.supports_len +=1
            else:
                if supports is None:
                    self.supports = []
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(device)
                self.supports_len += 1




        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1,kernel_size),dilation=new_dilation))

                self.gate_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *=2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(gcn(dilation_channels,residual_channels,dropout,support_len=self.supports_len))



        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1,1),
                                  bias=True)

        # self.end_conv_interm = nn.Conv2d(in_channels=dilation_channels * 10,
        #                           out_channels=end_channels,
        #                           kernel_size=(1,1),
        #                           bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=True)
        
        # self.conv1_reg = torch.nn.Conv1d(1, 8, 10, stride=1)  # .to(device)
        # self.conv2_reg = torch.nn.Conv1d(8, 16, 10, stride=1)
        # self.bn1_reg = torch.nn.BatchNorm1d(8)
        # self.bn2_reg = torch.nn.BatchNorm1d(16)
        # self.embedding_dim = 100
        # self.dim_fc = 26336 #  127 * 207


        # self.bn3_reg = torch.nn.BatchNorm1d(self.embedding_dim)

        self.linear = nn.Linear(220,256)
        # self.linear2 = nn.Linear(256,256)
        # self.linear3 = nn.Linear(220,256)
        self.receptive_field = receptive_field
        # self.centrality_linear = nn.Parameter(F.softmax(torch.Tensor(centrality).to(device)).reshape(1,207,1), requires_grad=True)
        self.layer_norm = nn.LayerNorm([256, 207,1])
        # nn.init.constant_(self.centrality_linear, F.softmax(torch.Tensor(centrality)).reshape(1,207,1))
        # self.bn_final = nn.BatchNorm3d(256, device=device)
        # # Regularization
        # off_diag = np.ones([207, 207])

        # def encode_onehot(labels):
        #     classes = set(labels)
        #     classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
        #                     enumerate(classes)}
        #     labels_onehot = np.array(list(map(classes_dict.get, labels)),
        #                              dtype=np.int32)
        #     return labels_onehot

        # rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
        # rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
        # self.rel_rec = torch.FloatTensor(rel_rec).to(device)
        # self.rel_send = torch.FloatTensor(rel_send).to(device)
        # self.fc = torch.nn.Linear(self.dim_fc, self.embedding_dim)
        # self.fc_out = nn.Linear(self.embedding_dim * 2, self.embedding_dim)
        # self.fc_cat = nn.Linear(self.embedding_dim, 2)
        self.layer_norm_2 = nn.LayerNorm([256, 207,1])



    def forward(self, input):
        # x = input.transpose(1, 0).reshape(207, 1, -1)
        # x = self.conv1_reg(x)
        # x = F.relu(x)
        # x = self.bn1_reg(x)
        # # x = self.hidden_drop(x)
        # x = self.conv2_reg(x)
        # x = F.relu(x)
        # x = self.bn2_reg(x)
        # x = x.reshape(207, -1)
        # x = self.fc(x)
        # x = F.relu(x)
        # x = self.bn3_reg(x)
        # receivers = torch.matmul(self.rel_rec, x)
        # senders = torch.matmul(self.rel_send, x)
        # x = torch.cat([senders, receivers], dim=1)
        # x = torch.relu(self.fc_out(x))
        # x = self.fc_cat(x)
        # ret = x
        tmp = torch.clone(input)
        in_len = input.size(3)
        if in_len<self.receptive_field:
            x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))
        else:
            x = input
# NEWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
        x = self.start_conv(x)
        skip = 0

        # calculate the current adaptive adj matrix once per iteration
        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]

        # x += self.centrality_linear
        # x, _ = self.sa(x.view(64,207*13,32), x.view(64,207*13,32), x.view(64,207*13,32))
        # x = x.view(64, 32, 207, 13)
        # WaveNet layers
        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

            #(dilation, init_dilation) = self.dilations[i]

            #residual = dilation_func(x, dilation, init_dilation, i)
            residual = x
            # dilated convolution
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate

            # parametrized skip connection

            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip


            if self.gcn_bool and self.supports is not None:
                if self.addaptadj:
                    x = self.gconv[i](x, new_supports)
                else:
                    x = self.gconv[i](x,self.supports)
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]


            x = self.bn[i](x)
        x = F.relu(skip)
        x = self.layer_norm(x)
        ttt = self.temporal_attention(tmp).matmul(self.h_x_sp)
        concat = torch.cat((self.spatial_attention(tmp),ttt),1).view(64,207,220)
        linear_out = self.linear(concat).view(64,256,207).unsqueeze(-1)
        # linear_handy = self.linear_hand(1 - F.softmax(self.shotest_path.view(1, 207, 207)))
        x = x + self.t_we * self.layer_norm_2(linear_out)
        # torch.Size([64, 256, 207, 1])
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        # return x, ret.softmax(-1)[:, 0].clone().reshape(207, -1)
        return x

