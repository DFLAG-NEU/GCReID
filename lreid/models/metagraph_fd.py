import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from IPython import embed

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features # F
        self.out_features = out_features  # F' 
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        # torch.mm 矩阵相乘， h.shape: (N, in_features) , W.shape: (in_features, out_features)--->Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec) #合并e和zero_vec, adj>0的地方保存
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training) #论文中的aij
        h_prime = torch.matmul(attention, Wh) #tensorf乘法 
        # h_prime 输出特征 h' 

        # print("metagraph_fd.py------------------------------line36")
        # embed()

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout # 0.6

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)


def Truncated_initializer(m):
    # sample u1:
    size = m.size()
    u1 = torch.rand(size)*(1-np.exp(-2)) + np.exp(-2) # torch.rand(3,3)：返回3*3的矩阵，矩阵元素符合0-1之间均匀分布
    # torch.rand(size)=torch.rand(torch.size([1,2048])), 将这 1行2048列 的数组随机初始化, 服从[0, 1]区间的均匀分布 
    # sample u2:
    u2 = torch.rand(size)
    # sample the truncated gaussian ~TN(0,1,[-2,2]):
    z = torch.sqrt(-2*torch.log(u1)) * torch.cos(2*np.pi*u2)
    m.data = z

class GraphConvolution(nn.Module):
    def __init__(self, hidden_dim, sparse_inputs=False, act=nn.Tanh(), bias=True, dropout=0.6):
        super(GraphConvolution, self).__init__()
        self.active_function = act
        self.dropout_rate = dropout
        if dropout>0:
            self.dropout = nn.Dropout(p=dropout)
        self.sparse_inputs = sparse_inputs
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.W = nn.Parameter(torch.zeros(size=(hidden_dim, hidden_dim)))
        Truncated_initializer(self.W)
        if self.bias:
            self.b = nn.Parameter(torch.zeros(hidden_dim))
        else:
            self.b = None
        self.device = torch.device('cuda')

    def forward(self, inputs, adj):
        x = inputs
        x = self.dropout(x)
        node_size = adj.size(0)
        I = torch.eye(node_size, requires_grad=False).to(self.device)
        adj = adj + I # +I 避免只计算一个node的所有邻居的特征的加权和，而忽略该node自己的特征
        D = torch.diag(torch.sum(adj, dim=1, keepdim=False)) # 度矩阵
        adj = torch.matmul(torch.inverse(D), adj) # 邻接矩阵
        pre_sup = torch.matmul(x, self.W)
        output = torch.matmul(adj, pre_sup)

        if self.bias:
            output += self.b
        if self.active_function is not None:
            return self.active_function(output)
        else:
            return output

class MetaGraph_fd(nn.Module):
    def __init__(self, hidden_dim, input_dim, sigma=2.0, proto_graph_vertex_num=16, meta_graph_vertex_num=128):
        super(MetaGraph_fd, self).__init__()
        self.hidden_dim, self.input_dim, self.sigma = hidden_dim, input_dim, sigma
        adj_mlp = nn.Linear(hidden_dim, 1) 
        # nn.Linear()：用于设置网络中的全连接层
        # torch.nn.Linear(in_features, out_features. bias=True)
        # 从输入输出的张量的shape角度来理解，相当于一个输入为[batch_size, in_features]的张量变换成了[batch_size, out_features]的输出张量
        Truncated_initializer(adj_mlp.weight)
        nn.init.constant_(adj_mlp.bias, 0.1)
        # torch.nn.init.constant_(tensor, val) :用值val填充向量。

        gate_mlp = nn.Linear(hidden_dim, 1) # Linear(in_features=hidden_dim, out_features=1, bias=True)
        Truncated_initializer(gate_mlp.weight)
        nn.init.constant_(gate_mlp.bias, 0.1)

        self.softmax = nn.Softmax(dim=0)
        self.meta_graph_vertex_num = meta_graph_vertex_num # Vk个数:积累的知识图的节点个数,默认是64
        self.proto_graph_vertex_num = proto_graph_vertex_num #  Vs个数:一个batch中的 person个数,不是样本总数
        self.meta_graph_vertex = nn.Parameter(torch.rand(meta_graph_vertex_num, input_dim)) # Vk随机初始化的,  Nk*d
        # nn.Parameter:把这个函数理解为类型转换函数，将一个不可训练的类型Tensor转换成可以训练的类型parameter并将这个parameter绑定到这个module里面(net.parameter()中就有这个绑定的parameter
        self.distance = nn.Sequential(adj_mlp, nn.Sigmoid()) #As, nn.Sequential()将As参数化,使其可学习
        self.gate = nn.Sequential(gate_mlp, nn.Sigmoid()) # Ak,nn.Sequential()将Ak参数化,使其可学习
        self.device = torch.device('cuda')
        # self.meta_GCN = GraphConvolution(self.hidden_dim).to(self.device)
        self.att = GAT(nfeat=self.hidden_dim,nclass=self.input_dim, nhid=8,dropout=0.6, nheads=8,alpha=0.2).to(self.device)
        # GAT(nfeat=输入特征的维度F， nclass=输出特征的维度F')
        # model = GAT(nfeat=features.shape[1], # 节点个数
        #         nhid=args.hidden, # 隐藏层节点个数 8
        #         nclass=int(labels.max()) + 1, 
        #         dropout=args.dropout,  # int 0.6
        #         nheads=args.nb_heads, # int 8
        #         alpha=args.alpha) # Alpha for the leaky_relu.


        self.MSE = nn.MSELoss(reduce='mean')
        self.register_buffer('meta_graph_vertex_buffer', torch.rand(self.meta_graph_vertex.size(), requires_grad=False))

    def StabilityLoss(self, old_vertex, new_vertex):
        old_vertex = F.normalize(old_vertex) #归一化
        new_vertex = F.normalize(new_vertex)

        # return torch.mean(torch.log(1 + torch.exp(torch.sqrt(torch.sum((old_vertex-new_vertex).pow(2), dim=1, keepdim=False)))))
        return torch.mean(torch.sum((old_vertex-new_vertex).pow(2), dim=1, keepdim=False)) # torch.sum(dim=1) 按行相加

    def forward(self, inputs): # inputs = features 

        # 谁调用的forward
        # AKA的前向传播.
        # 图池化怎么实现,注意力怎么实现



        correlation_meta = self._correlation(self.meta_graph_vertex_buffer, self.meta_graph_vertex.detach()) # Vk尖和Vk的余弦相似性
        # correlation_meta = torch.mean(F.cosine_simlirity(self.meta_graph_vertex_buffer, self.meta_graph_vertex.detach()))

        self.meta_graph_vertex_buffer = self.meta_graph_vertex.detach() 
        # self.meta_graph_vertex.shape : torch.Size([64, 2048])

        batch_size = inputs.size(0)//2
        
        # TKG节点 每个batch的输入特征
        # protos = inputs # 一个batch的特征 inputs.shape : torch.Size([16, 2048])   Vs∈Nb×d

        # meta_graph是带权邻接矩阵（Ak） 全连接
        # meta_graph = self._construct_graph(self.meta_graph_vertex, self.meta_graph_vertex).to(self.device) # AKG

        # 普通全连接，无向图，邻接矩阵只有0和1
        # 构建AKG
        meta_graph = self._construct_graph_samegraph(self.meta_graph_vertex, self.meta_graph_vertex).to(self.device)

        # meta_graph.shape : torch.Size([64, 64])

        # proto_graph是带权邻接矩阵（As）
        # proto_graph = self._construct_graph(protos, protos).to(self.device) 
        # 构建TKG
        protos = inputs
        proto_graph = self._construct_graph_samegraph(protos, protos).to(self.device)
        
        # proto_graph.shape : torch.Size([16, 16])

        m, n = protos.size(0), self.meta_graph_vertex.size(0) 
        # m = 16, n = 64

        # print("metagraph_fd.py---------------line 103")
        # embed()
    
        '''
        xx = torch.pow(protos, 2).sum(1, keepdim=True).expand(m,n) # torch.pow(tensor([2]),3)=tensor([8])
        
        yy = torch.pow(self.meta_graph_vertex, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist.addmm_(mat1=protos, mat2=self.meta_graph_vertex.t(), beta=1, alpha=-2) # 算两个矩阵的欧几里得距离
        dist_square = dist.clamp(min=1e-6) # 跨图距离
        cross_graph = self.softmax(
            (- dist_square / (2.0 * self.sigma))).to(self.device)
        
        super_garph = torch.cat((torch.cat((proto_graph, cross_graph),  dim=1), torch.cat((cross_graph.t(), meta_graph), dim=1)), dim=0)
        '''

        # 知识迁移的准备工作
        # 将跨图的邻接矩阵，全1
        cross_graph = self._construct_graph_crossgraph(protos, self.meta_graph_vertex).to(self.device)
        # 邻接矩阵
        super_garph = torch.cat((torch.cat((proto_graph, cross_graph),  dim=1), torch.cat((cross_graph.t(), meta_graph), dim=1)), dim=0)
        # super_graph.shape :torch.Size([80, 80])
                
        # features是super_graph图中所有顶点
        feature = torch.cat((protos, self.meta_graph_vertex), dim=0).to(self.device)
        # feature.shape : torch.Size([80, 2048])

        # representation 是VG
        # 第一层图卷积
        # representation = self.meta_GCN(feature, super_garph) #meta_GCN(inputs, adj)　VG
        # representation.shape : torch.Size([80, 2048])

         # 第二层图卷积
        # representation = self.meta_GCN(representation, super_garph)
        
        # most_similar_index = torch.argmax(cross_graph, dim=1)
        

        # GAT
        representation = self.att(feature, super_garph)
        
        # correlation_transfer_meta = self._correlation(representation[batch_size:].detach(), self.meta_graph_vertex.detach())
        correlation_transfer_meta = self._correlation(representation[-batch_size:].detach(), self.meta_graph_vertex.detach())
        # self._correlation(torch.Size([64, 2048]), torch.Size([64, 2048]))
        
        # correlation_protos = self._correlation(representation[0:batch_size].detach(), protos.detach())
        correlation_protos = self._correlation(representation[0:2*batch_size].detach(), protos.detach())
        # self._correlation(torch.Size([16, 2048]), torch.Size([16, 2048]))
        
        # print("matagraph_fd.py -------------------line259")
        # embed()
        
        # return representation[0:batch_size].to(self.device), representation[-batch_size:].to(self.device), [correlation_meta,correlation_transfer_meta, correlation_protos]
        
        return representation[0:batch_size].to(self.device), representation[batch_size:2*batch_size].to(self.device),representation[-batch_size:].to(self.device), [correlation_meta,correlation_transfer_meta, correlation_protos]
        # return 更新后的:增强样本/原始样本/积累知识节点
        # return representation[0:batch_size].to(self.device), [correlation_meta,correlation_transfer_meta, correlation_protos]

    def _construct_graph(self, A, B):
        # 构建全连接图.

        # A.size() --> torch.size([64, 2048])  (Vk节点个数, 每个节点的维度)
        m = A.size(0) # 32
        n = B.size(0) # 32
        I = torch.eye(n, requires_grad=False).to(self.device) # 单位阵
        # index_aabb和index_abab都是一维的torch.Size([4096])
        # index_aabb是[0,0,0,...,1,1,1,...,2,2,2,...,63,63,...]
        #index_abab是[0,1,2,3,...,62,63, 0,1,2,...,62,63,0,1,2,...]
        index_aabb = torch.arange(0, m, requires_grad=False).repeat_interleave(n, dim=0).long()
        index_abab = torch.arange(0, n, requires_grad=False).repeat(m).long()
        diff = A[index_aabb] - B[index_abab]
        graph = self.distance(diff).view(m, n) # view相当于reshape
        graph = graph.to(self.device) * (1 - I) + I
        return graph #返回值是邻接矩阵

    def _construct_graph_samegraph(self, A, B):
        # 定义普通的全连接邻接矩阵,除了对角线，剩下全是1
        m = A.size(0) # 32
        n = B.size(0) # 32
        I = torch.eye(n, requires_grad=False).to(self.device) # 单位阵
        graph=torch.tensor(np.ones((m,n))).to(self.device)-I 
        # graph = self.distance(graph)
        return graph
        

    def _construct_graph_crossgraph(self, A, B):
        # 定义普通的全连接邻接矩阵,剩下全是1
        m = A.size(0) # 32
        n = B.size(0) # 32
        graph=torch.tensor(np.ones((m,n))).to(self.device)
        # graph = self.distance(graph)
        return graph


    def _correlation(self, A, B):
        similarity = F.cosine_similarity(A,B)
        similarity = torch.mean(similarity) # 取平均 means(tensor([1,2,3,4])) = 2.5
        return similarity



class FixedMetaGraph(nn.Module):
    def __init__(self, hidden_dim, input_dim, sigma=2.0, proto_graph_vertex_num=16, meta_graph_vertex_num=128):
        super(FixedMetaGraph, self).__init__()
        self.hidden_dim, self.input_dim, self.sigma = hidden_dim, input_dim, sigma
        adj_mlp = nn.Linear(hidden_dim, 1)
        Truncated_initializer(adj_mlp.weight)
        nn.init.constant_(adj_mlp.bias, 0.1)

        gate_mlp = nn.Linear(hidden_dim, 1)
        Truncated_initializer(gate_mlp.weight)
        nn.init.constant_(gate_mlp.bias, 0.1)

        self.softmax = nn.Softmax(dim=0)
        self.meta_graph_vertex_num = meta_graph_vertex_num
        self.proto_graph_vertex_num = proto_graph_vertex_num
        self.meta_graph_vertex = nn.Parameter(torch.rand(meta_graph_vertex_num, input_dim), requires_grad=False)
        self.distance = nn.Sequential(adj_mlp, nn.Sigmoid())
        self.gate = nn.Sequential(gate_mlp, nn.Sigmoid())
        self.device = torch.device('cuda')
        self.meta_GCN = GraphConvolution(self.hidden_dim).to(self.device)
        self.MSE = nn.MSELoss(reduce='mean')



    def forward(self, inputs):
        batch_size = inputs.size(0)
        protos = inputs
        meta_graph = self._construct_graph(self.meta_graph_vertex, self.meta_graph_vertex).to(self.device)
        proto_graph = self._construct_graph(protos, protos).to(self.device)
        m, n = protos.size(0), self.meta_graph_vertex.size(0)
        xx = torch.pow(protos, 2).sum(1, keepdim=True).expand(m,n)
        yy = torch.pow(self.meta_graph_vertex, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist.addmm_(mat1=protos, mat2=self.meta_graph_vertex.t(), beta=1, alpha=-2)
        dist_square = dist.clamp(min=1e-6)
        cross_graph = self.softmax(
            (- dist_square / (2.0 * self.sigma))).to(self.device)
        super_garph = torch.cat((torch.cat((proto_graph, cross_graph),  dim=1), torch.cat((cross_graph.t(), meta_graph), dim=1)), dim=0)
        feature = torch.cat((protos, self.meta_graph_vertex), dim=0).to(self.device)
        representation = self.meta_GCN(feature, super_garph)
        # representation = self.att(feature, super_garph)

        # most_similar_index = torch.argmax(cross_graph, dim=1)
        normalized_transfered_meta = F.normalize(representation[batch_size:])
        normalized_meta = F.normalize(self.meta_graph_vertex)
        ccT = torch.mm(normalized_transfered_meta, normalized_transfered_meta.t())
        mmT = torch.mm(normalized_meta, normalized_meta.t())
        # I = torch.eye(self.meta_graph_vertex_num, requires_grad=False).to(self.device)
        correlation = self.MSE(ccT, mmT)

        return representation[0:batch_size].to(self.device), correlation

    def _construct_graph(self, A, B):
        m = A.size(0)
        n = B.size(0)
        I = torch.eye(n, requires_grad=False).to(self.device)
        index_aabb = torch.arange(0, m, requires_grad=False).repeat_interleave(n, dim=0).long()
        index_abab = torch.arange(0, n, requires_grad=False).repeat(m).long()
        diff = A[index_aabb] - B[index_abab]
        graph = self.distance(diff).view(m, n)
        graph = graph.to(self.device) * (1 - I) + I
        return graph

