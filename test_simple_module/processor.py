#!/usr/bin/env python
from __future__ import print_function

import os
import sys
import time
import json
import argparse
import subprocess
from importlib import import_module
from PhysicsTools.NanoAODTools.postprocessing.framework.postprocessor import PostProcessor
sys.path.append('.')
from TestQCDSampleProducer import *

import ROOT
import logging
import shutil
ROOT.PyConfig.IgnoreCommandLineOptions = True

def xrd_prefix(filepaths):
    prefix = ''
    allow_prefetch = False
    if not isinstance(filepaths, (list, tuple)):
        filepaths = [filepaths]
    filepath = filepaths[0]
    if filepath.startswith('/eos/cms'):
        prefix = 'root://eoscms.cern.ch/'
    elif filepath.startswith('/eos/user'):
        prefix = 'root://eosuser.cern.ch/'
    elif filepath.startswith('/eos/uscms'):
        prefix = 'root://cmseos.fnal.gov/'
    elif filepath.startswith('/store/'):
        # remote file
        import socket
        host = socket.getfqdn()
        if 'cern.ch' in host:
            prefix = 'root://xrootd-cms.infn.it//'
        else:
            prefix = 'root://cmsxrootd.fnal.gov//'
        allow_prefetch = True
    expanded_paths = [(prefix + '/' + f if prefix else f) for f in filepaths]
    return expanded_paths, allow_prefetch

def main(args):

    # cleaning
    for f in os.listdir('.'):
        if f.endswith('.root'):
            logging.warning("!!!Warning!!!: %s in the working dir, will cause overlap, please remove" %f)
            return False

    modules = [QCDTree_2017()]
    # run postprocessor
    inputfiles = args.files if len(args.files) else print("Please direct to Inputfiles")
    filepaths, allow_prefetch = xrd_prefix(inputfiles)
    print(filepaths)
    p = PostProcessor(outputDir='.',
                      inputFiles=filepaths,
                      cut="Sum$((Jet_pt>25 && abs(Jet_eta)<2.4 && (Jet_jetId & 2)) * Jet_pt)>200 && nFatJet>0",
                      branchsel='./keep_and_drop_input.txt',
                      modules=modules,
                      compression='LZ4:4',
                      firstEntry=0,
                      outputbranchsel='./keep_and_drop_output.txt'
                      )
    p.run()

    outputname = args.outputname
    print("outputname: ", outputname)
    # hadd files
    p = subprocess.Popen('haddnano.py %s *.root' % outputname, shell=True)
    p.communicate()
    if p.returncode != 0:
        raise RuntimeError('Hadd failed!')

    # stage out
    outputdir = args.outputdir
    print(outputdir)
    if outputdir.startswith('/eos'):
        cmd = 'xrdcp --silent -p -f {outputname} {outputdir}/{outputname}'.format(
            outputname=outputname, outputdir=xrd_prefix(outputdir)[0][0])
        print(cmd)
        success = False
        for count in range(3):
            p = subprocess.Popen(cmd, shell=True)
            p.communicate()
            if p.returncode == 0:
                success = True
                break
            else:
                time.sleep(args.sleep)
        if not success:
            raise RuntimeError("Stage out FAILED!")

        # clean up
        os.remove(outputname)

    else:
        if not os.path.exists("%s/test_output" %(outputdir)):
            os.mkdir("%s/test_output" %(outputdir))
        shutil.copyfile(outputname, "%s/test_output/%s" %(outputdir,outputname))

    # clean files
    for f in os.listdir('.'):
        if f.endswith('.root'):
            os.remove(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NanoAOD postprocessing.')
    parser.add_argument('--sleep',
                        type=int, default=120,
                        help='Seconds to wait before retry stageout. Default: %(default)s'
                        )
    parser.add_argument('--files',
                        nargs='*', default=[],
                        help='Run over the specified input file. Default:%(default)s')
    parser.add_argument('--outputdir',
                        type=str, default='./',
                        help='Run over the specified output file. Default:%(default)s')
    parser.add_argument('--outputname',
                        type=str, default='test.root',
                        help='output file name. Default:%(default)s')

    args = parser.parse_args()
    main(args)
