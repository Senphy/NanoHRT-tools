from PhysicsTools.NanoAODTools.postprocessing.framework.datamodel import Collection

import sys
sys.path.append('.')
from TestBaseProducer import TestBaseProducer


class TestSampleProducer(TestBaseProducer):

    def __init__(self, **kwargs):
        super(TestSampleProducer, self).__init__(channel='qcd', **kwargs)

    def beginFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        super(TestSampleProducer, self).beginFile(inputFile, outputFile, inputTree, wrappedOutputTree)

    def analyze(self, event):
        """process event, return True (go to next module) or False (fail, go to next event)"""

        self.selectLeptons(event)
        self.correctJetsAndMET(event)

        if len(event.fatjets) < 2:
            return False
        probe_jets = event.fatjets[:2]

        if self._opts['sfbdt_threshold'] > -99:
            self.selectSV(event)
            if len(event.secondary_vertices) < 2:
                return False

            for fj in probe_jets:
                if not (len(fj.subjets) == 2 and fj.msoftdrop > 50 and fj.msoftdrop < 200):
                    fj.is_qualified = False
                    continue

                self.matchSVToFatJets(event, [fj])
                if fj.sfBDT < self._opts['sfbdt_threshold']:
                    fj.is_qualified = False
                    continue

            if probe_jets[0].is_qualified is False and probe_jets[1].is_qualified is False:
                return False

        self.loadGenHistory(event, probe_jets)
        # self.evalTagger(event, probe_jets)
        # self.evalMassRegression(event, probe_jets)

        # fill output branches
        self.fillFatJetInfo(event, probe_jets)

        return True


# define modules using the syntax 'name = lambda : constructor' to avoid having them loaded when not needed
def QCDTree_2016(): return TestSampleProducer(year=2016)
def QCDTree_2017(): return TestSampleProducer(year=2017)
def QCDTree_2018(): return TestSampleProducer(year=2018)
