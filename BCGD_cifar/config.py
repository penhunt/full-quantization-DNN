load_float_model = 1
device = 'cuda'
binary = 'mean'
arch = 'vgg11'

batch_size = 128
lr = 0.1
weight_decay = 1e-4
momentum = 0.95

n_epochs = 200
m_epochs = 150
steps = [80, 140]

# rho = 1e-5*0
rho_rate = 1
eta_rate = 1.02

stage = 0
rate_factor = 0.01
initial_alpha = 0
gamma = 0
gd_type = 'mean'

quant = 1
bit = 4
ffun = 'quant'
bfun = 'inplt'