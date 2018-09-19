import torch
from ugans.core import Map

from IPython import embed


class SimGD(Map):
    def __init__(self,manager):
        super(SimGD, self).__init__(manager)

    def map(self, real_data, fake_data, data_att, atts, V=None, norm_d=None, norm_g=None):
        # 1. Get model outputs: (latents, attributes, d_dis_preds, d_probs)
        real_outputs, fake_outputs = self.m.get_outputs([real_data, fake_data])

        # 2. Define and record GAN losses
        V_real, V_fake, V_fake_g = self.m.get_V(self.m.params['batch_size'], real_outputs[-1], fake_outputs[-1])
        Vsum = V_real + V_fake
        V_d = -Vsum
        V_g = V_fake_g

        # 3. Record disentangling losses
        dis_error = self.m.att_loss(real_outputs[2], real_outputs[1]) + self.m.att_loss(fake_outputs[2], fake_outputs[1])

        # 4. Record attribute extractor losses
        att_error = self.m.att_loss(self.m.F_att(data_att), atts)

        # 5. Compute gradients
        map_g = torch.autograd.grad(V_g, self.m.G.parameters(), create_graph=True)
        map_att = torch.autograd.grad(att_error, self.m.F_att.parameters(), create_graph=True)
        map_lat = torch.autograd.grad(V_d - 0.1*dis_error, self.m.F_lat.parameters(), create_graph=True)
        map_d = torch.autograd.grad(V_d, self.m.D.parameters(), create_graph=True)
        map_dis = torch.autograd.grad(dis_error, self.m.D_dis.parameters(), create_graph=True)
        norm_d = sum([torch.sum(g**2.) for g in map_d])
        norm_g = sum([torch.sum(g**2.) for g in map_g])

        return [map_g, map_att, map_lat, map_d, map_dis, Vsum, norm_d, norm_g]
