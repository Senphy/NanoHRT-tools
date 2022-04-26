import logging
import os,sys
import logging
import shutil
import subprocess

def main(filepath):
    
    logging.getLogger().setLevel(logging.INFO)
    try:
        files = os.listdir(filepath)
    except:
        logging.info("Not a valid file path")
        return False
    
    tarball_path = "%s/../CMSSW.tar.gz" %(os.environ["CMSSW_BASE"])
    if not os.path.exists(tarball_path):
        logging.info("tarball not exists in %s, skipping" %(tarball_path))
        return False

    test_path = "%s/src/PhysicsTools/NanoHRTTools/test_simple_module/" %(os.environ["CMSSW_BASE"])

    if not os.path.exists("job_test"):
        os.makedirs("job_test/logs")
    else:
        shutil.rmtree("job_test")
        os.makedirs("job_test/logs")

    for file in files:

        if not file.endswith(".root"):
            logging.info("%s is not a valid root file, skipping" %(file))
            continue

        logging.info("Preparing submit file for %s" %(file))
        abbre_name = file.rstrip('.root')
        with open ("job_test/submit_%s.cmd" %(abbre_name), "w+" ) as f:
            f.write("universe \t = vanilla\n")
            f.write("executable \t = %s/src/PhysicsTools/NanoHRTTools/test_simple_module/run_test.sh\n" %(os.environ["CMSSW_BASE"]))
            f.write("arguments \t = %s/%s %s %s\n" %(str(filepath), str(file), test_path, str(file)))
            f.write("requirements \t = (Arch == \"X86_64\") && (OpSys == \"LINUX\")\n")
            f.write("+JobFlavour \t = testmatch\n\n")
            # f.write("request_cpus \t = 4\n")
            f.write("request_memory \t = 4096\n")
            f.write("request_disk \t = 10000000\n\n")
            f.write("error \t = %s/job_test/logs/%s.err \n" %(test_path, abbre_name))
            f.write("output \t = %s/job_test/logs/%s.out \n" %(test_path, abbre_name))
            f.write("log \t = %s/job_test/logs/%s.log \n" %(test_path, abbre_name))
            f.write("should_transfer_files \t = YES\n")
            f.write("transfer_input_files \t = \
                    %s, %s/keep_and_drop_input.txt, %s/keep_and_drop_output.txt,\
                    %s/TestBaseProducer.py, %s/TestQCDSampleProducer.py, %s/processor.py\n"\
                    %(tarball_path, test_path, test_path, test_path, test_path, test_path))
            # f.write("transfer_output_remaps \t = \"test.root = "+datasetname+"_file"+filename+".root\"\n")
            f.write("when_to_transfer_output \t = ON_EXIT\n")
            f.write("+MaxRuntime = 49600\n")
            f.write("queue 1")
        f.close()

        cmd = "condor_submit job_test/submit_{abbre_name}.cmd".format(abbre_name=abbre_name)
        p = subprocess.Popen(cmd, shell=True)
        p.communicate()
        if p.returncode == 0:
            success = True
            logging.info("Submit succeed for %s\n" %(file))
        else:
            logging.warning("Submit fail for %s\n" %(file))

if __name__ == '__main__':
    main("/eos/user/s/sdeng/sfbdt/output_test_slim_ak8_qcd_2017/mc/")