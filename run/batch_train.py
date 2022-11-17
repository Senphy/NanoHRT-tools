import multiprocessing
import os,sys
import subprocess
import math
import numpy as np
import timeit

x = 1
date = '20221106'

# designs = [
#     {
#         'folder_name' : 'tau31_1_3',
#         'sigexpr': f'(fj_{x}_matchedInitHadsTau3/fj_{x}_matchedInitHadsTau1)<0.1',
#         'bkgexpr': f'(fj_{x}_matchedInitHadsTau3/fj_{x}_matchedInitHadsTau1)>0.3',
#     },
# ]

def train(design):
    proc = f'python -u train_ulnanov9.py --model-dir {date}/{design["folder_name"]}_fj{x}_model/std --jet-idx {x} -s "{design["sigexpr"]}" -b "{design["bkgexpr"]}" --train --gpu 3'
    r=subprocess.run(args=proc,shell=True,stdout=subprocess.PIPE,encoding='utf-8')
    with open(f'{date}/{design["folder_name"]}_fj{x}_model/{design["folder_name"]}_fj{x}_model.log', 'w+') as f:
        f.write(r.stdout)

def infer(design):
    proc = f'for IDX in {x}; do \
python -u infer_ulnanov9.py --model-dir {date}/{design["folder_name"]}_fj{x}_model/std --predict --gpu 3 \
-i /home/pku/sdeng/sfbdt/test_output/ \
-o /home/pku/sdeng/sfbdt/{date}/{design["folder_name"]}_fj${{IDX}}_sfBDT \
--jet-idx ${{IDX}} --bdt-varname fj${{IDX}}_sfBDT_add; done'
    print(proc)
    r=subprocess.run(args=proc,shell=True,stdout=subprocess.PIPE,encoding='utf-8')
    with open(f'{date}/{design["folder_name"]}_fj{x}_sfBDT/{design["folder_name"]}_fj{x}_infer.log', 'w+') as f:
        f.write(r.stdout)

# -i /data/pku/home/licq/hcc/new/samples/trees_sf/20211128_ULNanoV9_ak8_qcd_2017/ \

if __name__ == '__main__':

    # for design in designs:
    #     design = designs[design]
    #     print(design['folder_name'])
    #     print(f'python -u train_ulnanov9.py --model-dir {date}/{design["folder_name"]}_fj{x}_model --jet-idx {x} -s "{design["sigexpr"]}" -b "{design["bkgexpr"]}" --train --gpu 3')


    designs = []
    total = 0
    for sig in np.arange(0.05, 0.35, 0.05):
        for bkg in np.arange(sig, max(0.45, sig+0.05), 0.05):

            # print(sig, bkg)
            # print(math.modf(sig), math.modf(bkg))
            # print(int(100*math.modf(sig)[0]), int(100*math.modf(bkg)[0]))
            design = {}
            design['folder_name'] = f'tau21_{round(100*math.modf(sig)[0])}_{round(100*math.modf(bkg)[0])}'
            design['sigexpr'] = f'(fj_{x}_matchedInitHadsTau2/fj_{x}_matchedInitHadsTau1)<{round(sig,2)}'
            design['bkgexpr'] = f'(fj_{x}_matchedInitHadsTau2/fj_{x}_matchedInitHadsTau1)>{round(bkg,2)}'
            print(design)
            print(total)
            designs.append(design)
            total+=1
    
    # infer(designs[0])
    p = multiprocessing.Pool(4)
    start = timeit.default_timer()
    # b = p.map(train, designs)
    b = p.map(infer, designs)
    p.close()
    p.join()
    end = timeit.default_timer()
    print('total train time:', str(end-start), ' s')


