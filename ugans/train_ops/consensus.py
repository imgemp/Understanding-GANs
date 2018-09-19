import torch
from ugans.core import Map

from IPython import embed


class Consensus(Map):
    def __init__(self,manager):
        super(Consensus, self).__init__(manager)

    def map(self, map_g, map_att, map_lat, map_d, map_dis, Vsum, norm_d, norm_g):
        # 1. Compute squared norm of gradient and differentiate
        norm = 0.5*(norm_d+norm_g)
        # if discriminator last layer is linear and div is Wasserstein then the Discriminator bias
        # (constant weight vector) disappears from minimax objective so indicate allow_unused=True
        norm_d_grad = torch.autograd.grad(norm, self.m.D.parameters(), create_graph=True, allow_unused=True)
        norm_g_grad = torch.autograd.grad(norm, self.m.G.parameters(), create_graph=True)
        gammaJTF_d = [self.m.params['gamma']*g if g is not None else 0*p for g,p in zip(norm_d_grad, self.m.D.parameters())]
        gammaJTF_g = [self.m.params['gamma']*g for g in norm_g_grad]

        # 2. Sum terms
        map_d = [a+b for a, b in zip(map_d, gammaJTF_d)]
        map_g = [a+b for a, b in zip(map_d, gammaJTF_g)]
        
        return [map_g, map_att, map_lat, map_d, map_dis, Vsum, norm_d, norm_g]
