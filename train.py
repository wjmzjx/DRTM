'''
start Training
'''
import numpy as np

import context_overlap as coo

from utils import *
from model import *


c_file = 'data/example.txt'
e_file = 'data/example.embedding'

extra_info, wd_m, dictionary = coo.cal_extra_info(c_file, e_file)
print(len(dictionary))
CO = coo.cal_co(c_file, 1)
pmi_m = coo.cal_pmi(c_file,h_pmi=0)+CO
d_doc = coo.cal_d_doc(c_file, e_file)
tmp_folder = 'outputs' 

model = DRTM(
    D = wd_m,
    A = pmi_m,
    B = extra_info,
    S = d_doc,
    alpha=1.0, 
    beta=3.0, 
    miu=0.1,
    n_topic=10, 
    max_iter=1000, 
    max_err=1e-3,
    gamma= 0.1,
    rho=0.5,
    fix_seed=True)



model.save_format(Hfile=tmp_folder+'/H_t{} a={} b={} m={} g={} r={}'.format(model.n_topic, model.alpha, model.beta,model.miu,model.gamma, model.rho))
model.save_topics(Tfile=tmp_folder+'/t{} a={} b={} m={} g={} r={}'.format(model.n_topic, model.alpha, model.beta,model.miu,model.gamma, model.rho), dictionary=dictionary)
