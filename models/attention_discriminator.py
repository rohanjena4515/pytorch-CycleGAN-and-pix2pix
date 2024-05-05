class AttnLayer(nn.Module):

    def __init__(self, channel_size):

        super().__init__()

        q_size = channel_size//4

        self.query = nn.Conv2d(channel_size, q_size, kernel_size=1)

        self.key = nn.Conv2d(channel_size, q_size, kernel_size=1)

        self.value = nn.Conv2d(channel_size, channel_size, kernel_size=1)

        self.q_size = q_size





    def forward(self, x):

        print("Attention layer")

        q, k, v = self.query(x), self.key(x), self.value(x)   # [b, c, h, w]

        print(x.shape, q.shape, k.shape, v.shape)

        attn = torch.einsum('bchw,bcst->bhwst', q, k) / self.q_size**0.5

        print(attn.shape)

        B, H, W, S, T = attn.shape

        attn = torch.nn.functional.softmax(attn.reshape(B, H*W, S, T), dim=1).reshape(B, H, W, S, T)

        value = torch.einsum('bhwst,bchw->bcst', attn, v)

        return value





class AttentionDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):

        super().__init__()

        net = [nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1), norm_layer(ndf), nn.LeakyReLU(0.2, True)]    # 128

        net += [nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1), norm_layer(ndf*2), nn.LeakyReLU(0.2, True)] # 64

        net += [nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1), norm_layer(ndf*4), nn.LeakyReLU(0.2, True)] # 32

        net.append(AttnLayer(ndf*4))

        net += [nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1), norm_layer(ndf*8), nn.LeakyReLU(0.2, True)]

        net.append(AttnLayer(ndf*8))  # 16

        net += [nn.Conv2d(ndf*8, ndf*16, kernel_size=4, stride=2, padding=1), norm_layer(ndf*16), nn.LeakyReLU(0.2, True)]  # 8

        net.append(AttnLayer(ndf*16))

        net += [nn.Conv2d(ndf*16, 1, kernel_size=1, stride=1, padding=0)]  # output 1 channel prediction map

        self.model = nn.Sequential(*net)



    def forward(self, input):

        return self.model(input)