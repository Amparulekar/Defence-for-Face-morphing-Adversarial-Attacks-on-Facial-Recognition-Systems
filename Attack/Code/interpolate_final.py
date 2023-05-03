from templates import *

device = 'cuda:0'
conf = ffhq128_autoenc_base()
# print(conf.name)
model = LitModel(conf)
state = torch.load('/content/diffae/checkpoints/ffhq128_autoenc_72M/checkpoints/ffhq128_autoenc_72M/last1.ckpt', map_location='cpu')
model.load_state_dict(state['state_dict'], strict=False)
model.ema_model.eval()
model.ema_model.to(device);

data = ImageDataset('imgs_interpolate', image_size=conf.img_size, exts=['jpg', 'JPG', 'png'], do_augment=False)
# print(data.shape)
batch = torch.stack([
    data[0]['img'],
    data[1]['img'],
    data[2]['img']
])#what is the form of data?

# import matplotlib.pyplot as plt
# plt.imshow(batch[0].permute([1, 2, 0]) / 2 + 0.5)

cond = model.encode(batch.to(device))
xT = model.encode_stochastic(batch.to(device), cond, T=250)

import numpy as np
def cos(a, b):
    a = a.view(-1)
    b = b.view(-1)
    a = F.normalize(a, dim=0)
    b = F.normalize(b, dim=0)
    return (a * b).sum()
alpha = torch.tensor(np.linspace(0, 1, 10, dtype=np.float32)).to(cond.device)

# #lin-sph
# intp = cond[0][None] * (1 - alpha[:, None]) + cond[1][None] * alpha[:, None]
# #what is [:none]???
# theta = torch.arccos(cos(xT[0], xT[1]))
# x_shape = xT[0].shape
# intp_x = (torch.sin((1 - alpha[:, None]) * theta) * xT[0].flatten(0, 2)[None] + torch.sin(alpha[:, None] * theta) * xT[1].flatten(0, 2)[None]) / torch.sin(theta)
# intp_x = intp_x.view(-1, *x_shape)
# print(cond[0][None].shape, xT[0].shape)

# # #sph_lin
# x_shape = xT[0].shape
# theta = torch.arccos(cos(cond[0][None], cond[1][None]))
# intp = (torch.sin((1 - alpha[:, None]) * theta) * cond[0][None] + torch.sin(alpha[:, None] * theta) * cond[1][None]) / torch.sin(theta)
# intp_x = xT[0].flatten(0, 2)[None] * (1 - alpha[:, None]) + xT[1].flatten(0, 2)[None]* alpha[:, None]
# intp_x = intp_x.view(-1, *x_shape)

# #lin-lin
# x_shape = xT[0].shape
# intp = cond[0][None] * (1 - alpha[:, None]) + cond[1][None] * alpha[:, None]
# intp_x = xT[0].flatten(0, 2)[None] * (1 - alpha[:, None]) + xT[1].flatten(0, 2)[None]* alpha[:, None]
# intp_x = intp_x.view(-1, *x_shape)

# #sph-sph
# x_shape = xT[0].shape
# theta = torch.arccos(cos(cond[0][None], cond[1][None]))
# intp = (torch.sin((1 - alpha[:, None]) * theta) * cond[0][None] + torch.sin(alpha[:, None] * theta) * cond[1][None]) / torch.sin(theta)
# theta = torch.arccos(cos(xT[0], xT[1]))
# intp_x = (torch.sin((1 - alpha[:, None]) * theta) * xT[0].flatten(0, 2)[None] + torch.sin(alpha[:, None] * theta) * xT[1].flatten(0, 2)[None]) / torch.sin(theta)
# intp_x = intp_x.view(-1, *x_shape)

# #quad-sph
# intp = torch.sqrt(torch.square(cond[0][None]) * (1 - alpha[:, None]) + torch.square(cond[1][None]) * alpha[:, None])
# print(intp.shape, (cond[0][None] * (1 - alpha[:, None])).shape)
# theta = torch.arccos(cos(xT[0], xT[1]))
# x_shape = xT[0].shape
# intp_x = (torch.sin((1 - alpha[:, None]) * theta) * xT[0].flatten(0, 2)[None] + torch.sin(alpha[:, None] * theta) * xT[1].flatten(0, 2)[None]) / torch.sin(theta)
# intp_x = intp_x.view(-1, *x_shape)

#lin-quad
# intp = cond[0][None] * (1 - alpha[:, None]) + cond[1][None] * alpha[:, None]
# x_shape = xT[0].shape
# intp_x = torch.sqrt(torch.square(xT[0].flatten(0, 2)[None]) * (1 - alpha[:, None]) + torch.square(xT[1].flatten(0, 2)[None])* alpha[:, None])
# intp_x = intp_x.view(-1, *x_shape)

#three people
import numpy as np
alpha1 = 0
alpha2 = 0

intp = cond[0][None] * (1 - alpha1 - alpha2) + cond[1][None] * alpha1 + cond[2][None] * alpha2

#considering cosine similarity as the average of the similarity between all three pairs
def cos(a, b, c):
    a = a.view(-1)
    b = b.view(-1)
    c = c.view(-1)
    a = F.normalize(a, dim=0)
    b = F.normalize(b, dim=0)
    c = F.normalize(c, dim=0)
    return ((a * b).sum()+(b * c).sum()+(a * c).sum())/3

theta = torch.arccos(cos(xT[0], xT[1], xT[2]))
x_shape = xT[0].shape
intp_x = (torch.sin((1 - alpha1 - alpha2) * theta) * xT[0].flatten(0, 2)[None] + torch.sin(alpha1 * theta) * xT[1].flatten(0, 2)[None] + torch.sin(alpha2 * theta) * xT[2].flatten(0, 2)[None]) / torch.sin(theta)
intp_x = intp_x.view(-1, *x_shape)

pred = model.render(intp_x, intp, T=20)
print(pred.type())
pred1 = pred.cpu().detach().numpy()
np.save('/content/diffae/saved/three_9.npy', pred1)


# import matplotlib.pyplot as plt
# torch.manual_seed(1)
# fig, ax = plt.subplots(1, 10, figsize=(5*10, 5))
# for i in range(len(alpha)):
    # ax[i].imshow(pred[i].permute(1, 2, 0).cpu())
# plt.savefig('imgs_manipulated/compare.png')