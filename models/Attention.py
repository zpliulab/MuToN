import torch
import torch.nn as nn

class Geo_attention(nn.Module):
    def __init__(
        self, Ni, Nd, Nh, radius=1.0,
    ):
        super(Geo_attention, self).__init__()

        # Transformation of the input features:
        self.Nh = Nh #number of Head
        self.Nd = Nd #dim of hidden
        self.Radius = torch.nn.Parameter(torch.tensor(radius).float(), requires_grad=False) #For distance smooth

        self.query_f = nn.Sequential(
            nn.Linear(Ni, Nd),
            nn.ELU(),
            nn.Linear(Nd, Nh*Nd),)

        self.geo_encoder = nn.Sequential(
            nn.Linear(12, Nd),
            nn.ELU(),
            nn.Linear(Nd, Nd),)

        self.key_f = nn.Sequential(
            nn.Linear(Nd, Nd),
            nn.ELU(),
            nn.Linear(Nd, Nd),)

        self.value_f = nn.Sequential(
            nn.Linear(Nd, Nd),
            nn.ELU(),
            nn.Linear(Nd, Nd),)

        self.decode_f = nn.Sequential(
            nn.Linear(Nh*Nd, Nd),
            nn.ELU(),
            nn.Linear(Nd, Nd),)
        self.layer_norm = nn.LayerNorm(Nd)
        # 3D convolution filters, encoded as an MLP:
        self.sdk = torch.nn.Parameter(torch.sqrt(torch.tensor(self.Nd).float()), requires_grad=False)


    def forward(self, features, x, nuv, topk):
        # features: [N, Nd]
        # x: [N, 3]
        # nuv: [N, 3, 3]
        # topk: [N, topk]

        #For query
        features_in = features
        N = features.shape[0] #number of residues
        nn = topk.shape[1] # number of neighbors of each residue
        Q = self.query_f(features).view(N, self.Nh, self.Nd) #[N, Nh, Nd]

        features_nn = features[topk] #[N, nn, Nd]
        x_nn = x[topk] #[N, nn, 3]
        orient_x = x_nn - x.unsqueeze(1) #[N, nn, 3]
        nuv_nn = nuv[topk] #[N, nn, 3, 3]
        orient_nuv = nuv_nn.view(N, nn*3, 3) #[N, nn*3, 3]
        dis = torch.norm(orient_x, dim=2, keepdim=True)
        dis = torch.exp(-torch.square(dis) / (2 * torch.square(self.Radius)))

        RL_x = torch.matmul(nuv, orient_x.transpose(1,2)).transpose(1,2) #[N, 3, 3] [N, 3, nn] -> [N, nn, 3] RL:Relative
        RL_nuv = torch.matmul(nuv, orient_nuv.transpose(1,2)).transpose(1,2) #[N, 3, 3] [N, 3, nn*3] -> [N, nn*3, 3]
        RL_nuv = RL_nuv.contiguous().view(N,nn,-1) #[N, nn, 3*3]
        RL = torch.cat([RL_x, RL_nuv], dim=-1) #[N, nn, 12]
        geo_fea = self.geo_encoder(RL)
        geo_fea = dis * geo_fea * features_nn

        #For Key
        #K = self.key_f(geo_fea).transpose(1,2)
        K = self.key_f(geo_fea).transpose(1,2)
        #For Value
        V = self.value_f(geo_fea)

        #coefficient
        Mq = torch.matmul(Q, K)
        mask = topk.unsqueeze(1) == 0
        Mq = Mq.masked_fill(mask, float('-inf'))
        Mq = torch.nn.functional.softmax(Mq / self.sdk, dim=2)  # [N, Nh, n]
        output = torch.matmul(Mq, V).view(N, self.Nh*self.Nd)
        output = self.decode_f(output)
        output = output + features_in
        return self.layer_norm(output)

class Interface_attention(nn.Module):
    def __init__(
        self, Ni, Nd, Nh, radius=1.0,
    ):
        super(Interface_attention, self).__init__()

        # Transformation of the input features:
        self.Nh = Nh #number of Head
        self.Nd = Nd
        self.Radius = torch.nn.Parameter(torch.tensor(radius).float(), requires_grad=False) #For distance smooth
        self.query_f = nn.Sequential(
            nn.Linear(Ni, Nd),
            nn.ELU(),
            nn.Linear(Nd, Nh*Nd),
        )

        self.geo_encoder = nn.Sequential(
            nn.Linear(12, Nd),
            nn.ELU(),
            nn.Linear(Nd, Nd),
        )

        self.key_f = nn.Sequential(
            nn.Linear(Nd, Nd, bias=False),
            nn.ELU(),
            nn.Linear(Nd, Nd, bias=False),
        )

        self.value_f = nn.Sequential(
            nn.Linear(Nd, Nd, bias=False),
            nn.ELU(),
            nn.Linear(Nd, Nd, bias=False))

        self.decode_f = nn.Sequential(
            nn.Linear(Nh*Nd, Nd, bias=False),
            nn.ELU(),
            nn.Linear(Nd, Nd, bias=False),
            nn.ELU(),
        )
        self.layer_norm = nn.LayerNorm(Nd)
        self.sdk = torch.nn.Parameter(torch.sqrt(torch.tensor(self.Nd).float()), requires_grad=False)

    def forward(self, features1, features2, x1, x2, nuv1, nuv2, topk):
        #features1, x1, nuv1, sourced from the wild one, N1
        #features2, x2, nuv2, sourced from the mutated one, N2
        #
        # x: [N, 3]
        # nuv: [N, 3, 3]
        # topk: [N, topk]
        N1 = x1.shape[0]
        N2 = x2.shape[0]
        nn = topk.shape[1]
        #For query
        Q = self.query_f(features1).view(N1, self.Nh, self.Nd) #N2*h*d

        features_nn = features2[topk] #N2*
        x_nn = x2[topk]
        orient_x = x_nn - x1.unsqueeze(1)
        nuv_nn = nuv2[topk]
        orient_nuv = nuv_nn.view(N1, nn*3, 3) #[N, n*3, 3]

        RL_x = torch.matmul(nuv1, orient_x.transpose(1,2)).transpose(1,2)
        RL_nuv = torch.matmul(nuv1, orient_nuv.transpose(1,2)).transpose(1,2) #[N, 3, 3] [N, 3, n*3] -> [N, n*3, 3]
        RL_nuv = RL_nuv.contiguous().view(N1,nn,-1)
        RL = torch.cat([RL_x, RL_nuv], dim=-1)
        # RL = RL * dis
        geo_fea = self.geo_encoder(RL)
        dis = torch.norm(orient_x, dim=2, keepdim=True)
        # mask_dis = dis > 5.0
        # Mq = torch.nn.functional.softmax(Mq / self.sdk, dim=2)  # [N1, Nh, n]
        dis = torch.exp(-torch.square(dis) / (2*torch.square(self.Radius)))

        geo_fea = geo_fea * features_nn * dis

        #For Key
        K = self.key_f(geo_fea).transpose(1,2)

        #For Value
        V = self.value_f(geo_fea)

        #coefficient
        Mq = torch.matmul(Q, K)
        mask = topk.unsqueeze(1) == 0
        Mq = Mq.masked_fill(mask, float(0.0))

        output = torch.matmul(Mq, V).view(N1, self.Nh*self.Nd)
        output = self.decode_f(output)
        return output

class Interface(nn.Module):
    def __init__(
        self, Ni, Nd, Nh, radius=1.0,
    ):
        super(Interface, self).__init__()

        # Transformation of the input features:
        self.Nh = Nh #number of Head
        self.Nd = Nd
        self.Radius = torch.nn.Parameter(torch.tensor(radius).float(), requires_grad=False) #For distance smooth
        self.query_f = nn.Sequential(
            nn.Linear(Ni*2, Nd),
            nn.ELU(),
            nn.Linear(Nd, Nd),
            nn.ELU()
        )

        self.layer_norm = nn.LayerNorm(Nd)
        self.sdk = torch.nn.Parameter(torch.sqrt(torch.tensor(self.Nd).float()), requires_grad=False)

    def forward(self, features1, features2, x1, x2, nuv1, nuv2, topk):
        #features1, x1, nuv1, sourced from the wild one, N1
        #features2, x2, nuv2, sourced from the mutated one, N2
        #
        # x: [N, 3]
        # nuv: [N, 3, 3]
        # topk: [N, topk]
        N1 = x1.shape[0]
        N2 = x2.shape[0]
        nn = topk.shape[1]
        #For query
        mask = topk.unsqueeze(-1) == 0
        features_nn = features2[topk] #N2*
        feature_self = features1.unsqueeze(1).expand(-1, nn, -1)
        features_nn = torch.cat([features_nn, feature_self], dim=-1)
        features_nn = self.query_f(features_nn)
        x_nn = x2[topk]
        orient_x = x_nn - x1.unsqueeze(1)
        dis = torch.norm(orient_x, dim=2, keepdim=True)
        dis = torch.exp(-torch.square(dis) / (2*torch.square(self.Radius)))
        dis = dis.masked_fill(mask, float(0.0)).transpose(1,2)
        features = torch.matmul(dis, features_nn).squeeze(1)
        #For Key

        return features

if __name__ == '__main__':
    import random
    f1 = torch.rand(100, 32)
    x1 = torch.rand(100, 3)
    nuv1 = torch.rand(100, 3, 3)
    idx = [0, 20, 50, 80, 100]
    topk = []

    for index in range(len(idx)-1):
        N = idx[index+1] - idx[index]
        topk_ = torch.LongTensor(random.sample(range(10000), 6*N)).view([N, 6])
        topk_ = topk_ % N + idx[index]
        topk.append(topk_)
    topk = torch.vstack(topk)
    model = Geo_attention(32, 32, 4, radius=12.0)
    a = model(f, x, nuv, topk)
    print('lalal')
    pass

    # import torch  # 引用torch
    # import matplotlib.pyplot as plt  # 引用matplotlib
    # import os  # 引用OS
    #
    # os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
    #
    # x_tor=torch.linspace(-5, 5, 200)  # 使用torch在区间[-5,5]上均匀的生成200个数据
    # x_np=x_tor.data.numpy()  # 将数据转换成np的数据类型
    # y_tanh=torch.tanh(x_tor).data.numpy()  # tanh激活函数
    # plt.plot(x_np, y_tanh, c='blue', label='tanh')  # 坐标轴赋值及描述信息
    # plt.ylim((-1.2, 1.2))  # 设置Y轴上下限
    # plt.legend(loc='best')  # 给图像加图例
    # plt.show()