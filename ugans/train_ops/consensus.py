import torch
from ugans.core import Map

from IPython import embed


class Consensus(Map):
    def __init__(self,manager):
        super(Consensus, self).__init__(manager)

    def map(self, mps, losses, norms):
        map_g, map_att, map_lat_gan, map_lat_dis, map_d, map_dis = mps
        Vsum, att_error, dis_error = losses
        norm_g, norm_att, norm_lat_gan, norm_lat_dis, norm_d, norm_dis = norms

        # 1. Compute squared norm of gradient and differentiate
        # if discriminator last layer is linear and div is Wasserstein then the Discriminator bias
        # (constant weight vector) disappears from minimax objective so indicate allow_unused=True
        if self.m.params['lat_dis_reg'] > 0:
            norm_dis = 0.5*(norm_lat_dis/(self.m.params['lat_dis_reg']**2)+norm_dis)
        else:
            norm_dis = 0.5*(norm_lat_dis+norm_dis)
        norm_dis_grad = torch.autograd.grad(norm_dis, self.m.D_dis.parameters(), create_graph=True)
        norm_lat_grad_dis = torch.autograd.grad(norm_dis, self.m.F_lat.parameters(), create_graph=True)
        norm_gan = 0.5*(norm_g+norm_d+norm_lat_gan)
        norm_d_grad = torch.autograd.grad(norm_gan, self.m.D.parameters(), create_graph=True, allow_unused=True)
        norm_g_grad = torch.autograd.grad(norm_gan, self.m.G.parameters(), create_graph=True)
        norm_lat_grad_gan = torch.autograd.grad(norm_gan, self.m.F_lat.parameters(), create_graph=True)

        gammaJTF_dis = [self.m.params['gamma_dis']*g for g in norm_dis_grad]
        gammaJTF_lat_dis = [self.m.params['lat_dis_reg']*self.m.params['gamma_dis']*g for g in norm_lat_grad_dis]
        gammaJTF_d = [self.m.params['gamma_v']*g if g is not None else 0*p for g,p in zip(norm_d_grad, self.m.D.parameters())]
        gammaJTF_g = [self.m.params['gamma_v']*g for g in norm_g_grad]
        gammaJTF_lat_gan = [self.m.params['gamma_v']*g for g in norm_lat_grad_gan]

        # 2. Sum terms
        map_dis = [a+b for a, b in zip(map_dis, gammaJTF_dis)]
        map_lat_dis = [a+b for a, b in zip(map_lat_dis, gammaJTF_lat_dis)]
        map_d = [a+b for a, b in zip(map_d, gammaJTF_d)]
        map_g = [a+b for a, b in zip(map_g, gammaJTF_g)]
        map_lat_gan = [a+b for a, b in zip(map_lat_gan, gammaJTF_lat_gan)]

        mps = [map_g, map_att, map_lat_gan, map_lat_dis, map_d, map_dis]
        
        return [mps, losses, norms]
