import os
import itertools
import numpy as np
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True

from PhysicsTools.NanoAODTools.postprocessing.framework.datamodel import Collection, Object
from PhysicsTools.NanoAODTools.postprocessing.framework.eventloop import Module

from PhysicsTools.NanoHRTTools.helpers.utils import deltaR, closest, polarP4, sumP4, get_subjets, corrected_svmass, configLogger
from PhysicsTools.NanoHRTTools.helpers.xgbHelper import XGBEnsemble
from PhysicsTools.NanoHRTTools.helpers.nnHelper import convert_prob, ensemble
from PhysicsTools.NanoHRTTools.helpers.jetmetCorrector import JetMETCorrector, rndSeed

import logging
logger = logging.getLogger('nano')
configLogger('nano', loglevel=logging.INFO)

lumi_dict = {2015: 19.52, 2016: 16.81, 2017: 41.48, 2018: 59.83}

class _NullObject:
    '''An null object which does not store anything, and does not raise exception.'''

    def __bool__(self):
        return False

    def __nonzero__(self):
        return False

    def __getattr__(self, name):
        pass

    def __setattr__(self, name, value):
        pass

class METObject(Object):

    def p4(self):
        return polarP4(self, eta=None, mass=None)

class TestBaseProducer(Module, object):

    def __init__(self, channel, **kwargs):
        self.jetType = 'ak8'
        self._channel = channel  # 'qcd', 'photon', 'inclusive', 'muon'
        self.year = int(kwargs['year'])
        self._jmeSysts = {'jec': False, 'jes': None, 'jes_source': '', 'jes_uncertainty_file_prefix': '',
                          'jer': None, 'jmr': None, 'met_unclustered': None, 'smearMET': True, 'applyHEMUnc': False}
        self._opts = {'sfbdt_threshold': -99,
                      'run_tagger': False, 'tagger_versions': ['V02b', 'V02c', 'V02d'],
                      'run_mass_regression': False, 'mass_regression_versions': ['V01a', 'V01b', 'V01c'],
                      'WRITE_CACHE_FILE': False}
        for k in kwargs:
            if k in self._jmeSysts:
                self._jmeSysts[k] = kwargs[k]
            else:
                self._opts[k] = kwargs[k]
        self._needsJMECorr = any([self._jmeSysts['jec'], self._jmeSysts['jes'],
                                  self._jmeSysts['jer'], self._jmeSysts['jmr'],
                                  self._jmeSysts['met_unclustered'], self._jmeSysts['applyHEMUnc']])

        logger.info('Running %s channel for %s jets with JME systematics %s, other options %s',
                    self._channel, self.jetType, str(self._jmeSysts), str(self._opts))

        self._fill_sv = True
        self._jetConeSize = 0.8
        self._fj_name = 'FatJet'
        self._sj_name = 'SubJet'
        self._fj_gen_name = 'GenJetAK8'
        self._sj_gen_name = 'SubGenJetAK8'
        self._sfbdt_files = [
            os.path.expandvars(
                '$CMSSW_BASE/src/PhysicsTools/NanoHRTTools/data/sfBDT/ak8_ul/xgb_train_qcd.model.%d' % idx)
            for idx in range(10)]  # FIXME: update to AK8 training
        self._sfbdt_vars = ['fj_2_tau21', 'fj_2_sj1_rawmass', 'fj_2_sj2_rawmass',
                            'fj_2_ntracks_sv12', 'fj_2_sj1_sv1_pt', 'fj_2_sj2_sv1_pt']

        # https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation
        self.DeepJet_WP_L = {2015: 0.0508, 2016: 0.0480, 2017: 0.0532, 2018: 0.0490}[self.year]
        self.DeepJet_WP_M = {2015: 0.2598, 2016: 0.2489, 2017: 0.3040, 2018: 0.2783}[self.year]
        self.DeepJet_WP_T = {2015: 0.6502, 2016: 0.6377, 2017: 0.7476, 2018: 0.7100}[self.year]

    def beginJob(self):
        pass

    def beginFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        self.isMC = bool(inputTree.GetBranch('genWeight'))
        self.hasParticleNetProb = bool(inputTree.GetBranch(self._fj_name + '_ParticleNetMD_probXbb'))
        self.out = wrappedOutputTree
        self.out.branch("test_branch", "F")
        # Large-R jets
        for idx in ([1, 2] if self._channel == 'qcd' else [1]):
            prefix = 'fj_%d_' % idx
            # matching variables
            if self._fill_sv:
                # last parton list
                if self.isMC:
                    for ptsuf in ['', '50']:
                        pass
                        for var in ['pt','eta','phi']:
                            self.out.branch(prefix + "np{pt}_{var}".format(pt=ptsuf, var=var), "F", lenVar=prefix + "npart")
                            self.out.branch(prefix + "gp{pt}_{var}".format(pt=ptsuf, var=var), "F", lenVar=prefix + "ngpart")
                            self.out.branch(prefix + "cp{pt}_{var}".format(pt=ptsuf, var=var), "F", lenVar=prefix + "ncpart")
                            self.out.branch(prefix + "bp{pt}_{var}".format(pt=ptsuf, var=var), "F", lenVar=prefix + "nbpart")
                            self.out.branch(prefix + "lp{pt}_{var}".format(pt=ptsuf, var=var), "F", lenVar=prefix + "nlpart")
                        # self.out.branch(prefix + "npart{}".format(ptsuf), "I")
                        # self.out.branch(prefix + "nbpart{}".format(ptsuf), "I")
                        # self.out.branch(prefix + "ncpart{}".format(ptsuf), "I")
                        # self.out.branch(prefix + "ngpart{}".format(ptsuf), "I")
        self.h_gp_bpart_2d = ROOT.TH2D('h_gp_bpart_2d', 'h_gp_bpart_2d', 20, -2.4, 2.4, 20, -3.2, 3.2)
        self.h_gp_cpart_2d = ROOT.TH2D('h_gp_cpart_2d', 'h_gp_cpart_2d', 20, -2.4, 2.4, 20, -3.2, 3.2)
        self.h_gp_gpart_2d = ROOT.TH2D('h_gp_gpart_2d', 'h_gp_gpart_2d', 20, -2.4, 2.4, 20, -3.2, 3.2)
        self.h_gp_npart_2d = ROOT.TH2D('h_gp_npart_2d', 'h_gp_npart_2d', 20, -2.4, 2.4, 20, -3.2, 3.2)
        self.h_gp_lpart_2d = ROOT.TH2D('h_gp_lpart_2d', 'h_gp_lpart_2d', 20, -2.4, 2.4, 20, -3.2, 3.2)
        self.h_gp_bpart50_2d = ROOT.TH2D('h_gp_bpart50_2d', 'h_gp_bpart50_2d', 20, -2.4, 2.4, 20, -3.2, 3.2)
        self.h_gp_cpart50_2d = ROOT.TH2D('h_gp_cpart50_2d', 'h_gp_cpart50_2d', 20, -2.4, 2.4, 20, -3.2, 3.2)
        self.h_gp_gpart50_2d = ROOT.TH2D('h_gp_gpart50_2d', 'h_gp_gpart50_2d', 20, -2.4, 2.4, 20, -3.2, 3.2)
        self.h_gp_npart50_2d = ROOT.TH2D('h_gp_npart50_2d', 'h_gp_npart50_2d', 20, -2.4, 2.4, 20, -3.2, 3.2)
        self.h_gp_lpart50_2d = ROOT.TH2D('h_gp_lpart50_2d', 'h_gp_lpart50_2d', 20, -2.4, 2.4, 20, -3.2, 3.2)

    def endFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        self.h_gp_bpart_2d.Write()
        self.h_gp_cpart_2d.Write()
        self.h_gp_gpart_2d.Write()
        self.h_gp_npart_2d.Write()
        self.h_gp_lpart_2d.Write()
        self.h_gp_bpart50_2d.Write()
        self.h_gp_cpart50_2d.Write()
        self.h_gp_gpart50_2d.Write()
        self.h_gp_npart50_2d.Write()
        self.h_gp_lpart50_2d.Write()
        pass

    def selectLeptons(self, event):
        # do lepton selection
        event.looseLeptons = []  # used for jet lepton cleaning & lepton counting

        electrons = Collection(event, "Electron")
        for el in electrons:
            el.etaSC = el.eta + el.deltaEtaSC
            if el.pt > 10 and abs(el.eta) < 2.5 and abs(el.dxy) < 0.05 and abs(el.dz) < 0.2 \
                    and el.mvaFall17V2noIso_WP90 and el.miniPFRelIso_all < 0.4:
                event.looseLeptons.append(el)

        muons = Collection(event, "Muon")
        for mu in muons:
            if mu.pt > 10 and abs(mu.eta) < 2.4 and abs(mu.dxy) < 0.05 and abs(mu.dz) < 0.2 \
                    and mu.looseId and mu.miniPFRelIso_all < 0.4:
                event.looseLeptons.append(mu)

        event.looseLeptons.sort(key=lambda x: x.pt, reverse=True)

    def correctJetsAndMET(self, event):
        # correct Jets and MET
        event.idx = event._entry if event._tree._entrylist is None else event._tree._entrylist.GetEntry(event._entry)
        event._allJets = Collection(event, "Jet")
        event.met = METObject(event, "MET")
        event._allFatJets = Collection(event, self._fj_name)
        event.subjets = Collection(event, self._sj_name)  # do not sort subjets after updating!!

        if self._needsJMECorr:
            rho = event.fixedGridRhoFastjetAll
            # correct AK4 jets and MET
            self.jetmetCorr.setSeed(rndSeed(event, event._allJets))
            self.jetmetCorr.correctJetAndMET(jets=event._allJets, lowPtJets=Collection(event, "CorrT1METJet"),
                                             met=event.met, rawMET=METObject(event, "RawMET"),
                                             defaultMET=METObject(event, "MET"),
                                             rho=rho, genjets=Collection(event, 'GenJet') if self.isMC else None,
                                             isMC=self.isMC, runNumber=event.run)
            event._allJets = sorted(event._allJets, key=lambda x: x.pt, reverse=True)  # sort by pt after updating

            # correct fatjets
            self.fatjetCorr.setSeed(rndSeed(event, event._allFatJets))
            self.fatjetCorr.correctJetAndMET(jets=event._allFatJets, met=None, rho=rho,
                                             genjets=Collection(event, self._fj_gen_name) if self.isMC else None,
                                             isMC=self.isMC, runNumber=event.run)
            # correct subjets
            self.subjetCorr.setSeed(rndSeed(event, event.subjets))
            self.subjetCorr.correctJetAndMET(jets=event.subjets, met=None, rho=rho,
                                             genjets=Collection(event, self._sj_gen_name) if self.isMC else None,
                                             isMC=self.isMC, runNumber=event.run)

        # jet mass resolution smearing
        if self.isMC and self._jmeSysts['jmr']:
            raise NotImplementedError

        # link fatjet to subjets and recompute softdrop mass
        for idx, fj in enumerate(event._allFatJets):
            fj.idx = idx
            fj.is_qualified = True
            fj.subjets = get_subjets(fj, event.subjets, ('subJetIdx1', 'subJetIdx2'))
            fj.msoftdrop = sumP4(*fj.subjets).M()
        event._allFatJets = sorted(event._allFatJets, key=lambda x: x.pt, reverse=True)  # sort by pt

        # select lepton-cleaned jets
        event.fatjets = [fj for fj in event._allFatJets if fj.pt > 200 and abs(fj.eta) < 2.4 and (
            fj.jetId & 2) and closest(fj, event.looseLeptons)[1] >= self._jetConeSize]
        event.ak4jets = [j for j in event._allJets if j.pt > 25 and abs(j.eta) < 2.4 and (
            j.jetId & 4) and closest(j, event.looseLeptons)[1] >= 0.4]
        event.ht = sum([j.pt for j in event.ak4jets])

    def selectSV(self, event):
        event._allSV = Collection(event, "SV")
        event.secondary_vertices = []
        for sv in event._allSV:
            # if sv.ntracks > 2 and abs(sv.dxy) < 3. and sv.dlenSig > 4:
            # if sv.dlenSig > 4:
            if True:
                event.secondary_vertices.append(sv)
        event.secondary_vertices = sorted(event.secondary_vertices, key=lambda x: x.pt, reverse=True)  # sort by pt
        # event.secondary_vertices = sorted(event.secondary_vertices, key=lambda x : x.dxySig, reverse=True)  # sort by dxysig

    def matchSVToFatJets(self, event, fatjets):
        # match SV to fatjets
        for fj in fatjets:
            fj.sv_list = []
            for sv in event.secondary_vertices:
                if deltaR(sv, fj) < self._jetConeSize:
                    fj.sv_list.append(sv)
            # match SV to subjets
            drcut = min(0.4, 0.5 * deltaR(*fj.subjets)) if len(fj.subjets) == 2 else 0.4
            for sj in fj.subjets:
                sj.sv_list = []
                for sv in event.secondary_vertices:
                    if deltaR(sv, sj) < drcut:
                        sj.sv_list.append(sv)

            fj.nsv_ptgt25 = 0
            fj.nsv_ptgt50 = 0
            fj.ntracks = 0
            fj.ntracks_sv12 = 0
            for isv, sv in enumerate(fj.sv_list):
                fj.ntracks += sv.ntracks
                if isv < 2:
                    fj.ntracks_sv12 += sv.ntracks
                if sv.pt > 25:
                    fj.nsv_ptgt25 += 1
                if sv.pt > 50:
                    fj.nsv_ptgt50 += 1

            # sfBDT & sj12_masscor_dxysig
            fj.sfBDT = -1
            fj.sj12_masscor_dxysig = 0
            if len(fj.subjets) == 2:
                sj1, sj2 = fj.subjets
                if len(sj1.sv_list) > 0 and len(sj2.sv_list) > 0:
                    sj1_sv, sj2_sv = sj1.sv_list[0], sj2.sv_list[0]
                    sfbdt_inputs = {
                        'fj_2_tau21': fj.tau2 / fj.tau1 if fj.tau1 > 0 else 99,
                        'fj_2_sj1_rawmass': sj1.mass,
                        'fj_2_sj2_rawmass': sj2.mass,
                        'fj_2_ntracks_sv12': fj.ntracks_sv12,
                        'fj_2_sj1_sv1_pt': sj1_sv.pt,
                        'fj_2_sj2_sv1_pt': sj2_sv.pt,
                    }
                    fj.sfBDT = self.xgb.eval(sfbdt_inputs, model_idx=(event.event % 10))
                    fj.sj12_masscor_dxysig = corrected_svmass(sj1_sv if sj1_sv.dxySig > sj2_sv.dxySig else sj2_sv)

    def loadGenHistory(self, event, fatjets):
        # gen matching
        if not self.isMC:
            return

        try:
            genparts = event.genparts
        except RuntimeError as e:
            genparts = Collection(event, "GenPart")
            for idx, gp in enumerate(genparts):
                if 'dauIdx' not in gp.__dict__:
                    gp.dauIdx = []
                if gp.genPartIdxMother >= 0:
                    mom = genparts[gp.genPartIdxMother]
                    if 'dauIdx' not in mom.__dict__:
                        mom.dauIdx = [idx]
                    else:
                        mom.dauIdx.append(idx)
            event.genparts = genparts

        def isHadronic(gp):
            if len(gp.dauIdx) == 0:
                return False
                # raise ValueError('Particle has no daughters!')
            for idx in gp.dauIdx:
                if abs(genparts[idx].pdgId) < 6:
                    return True
            return False

        def getFinal(gp):
            for idx in gp.dauIdx:
                dau = genparts[idx]
                if dau.pdgId == gp.pdgId:
                    return getFinal(dau)
            return gp

        lepGenTops = []
        hadGenTops = []
        hadGenWs = []
        hadGenZs = []
        hadGenHs = []

        for gp in genparts:
            if gp.statusFlags & (1 << 13) == 0:
                continue
            if abs(gp.pdgId) == 6:
                for idx in gp.dauIdx:
                    dau = genparts[idx]
                    if abs(dau.pdgId) == 24:
                        genW = getFinal(dau)
                        gp.genW = genW
                        if isHadronic(genW):
                            hadGenTops.append(gp)
                        else:
                            lepGenTops.append(gp)
                    elif abs(dau.pdgId) in (1, 3, 5):
                        gp.genB = dau
            elif abs(gp.pdgId) == 24:
                if isHadronic(gp):
                    hadGenWs.append(gp)
            elif abs(gp.pdgId) == 23:
                if isHadronic(gp):
                    hadGenZs.append(gp)
            elif abs(gp.pdgId) == 25:
                if isHadronic(gp):
                    hadGenHs.append(gp)

        for parton in itertools.chain(lepGenTops, hadGenTops):
            parton.daus = (parton.genB, genparts[parton.genW.dauIdx[0]], genparts[parton.genW.dauIdx[1]])
            parton.genW.daus = parton.daus[1:]
        for parton in itertools.chain(hadGenWs, hadGenZs, hadGenHs):
            parton.daus = (genparts[parton.dauIdx[0]], genparts[parton.dauIdx[1]])

        for fj in fatjets:
            fj.genH, fj.dr_H = closest(fj, hadGenHs)
            fj.genZ, fj.dr_Z = closest(fj, hadGenZs)
            fj.genW, fj.dr_W = closest(fj, hadGenWs)
            fj.genT, fj.dr_T = closest(fj, hadGenTops)
            fj.genLepT, fj.dr_LepT = closest(fj, lepGenTops)

        if self._fill_sv:
            # bb/cc matching
            # FIXME: only available for qcd & ggh(cc/bb) sample
            probe_fj = event.fatjets[1 if self._channel == 'qcd' else 0]
            probe_fj.genBhadron, probe_fj.genChadron = [], []
            for gp in genparts:
                if gp.pdgId in [5, -5] and gp.genPartIdxMother>=0 and genparts[gp.genPartIdxMother].pdgId in [21, 25] and deltaR(gp, probe_fj)<=self._jetConeSize:
                    if len(probe_fj.genBhadron)==0 or (len(probe_fj.genBhadron)>0 and gp.genPartIdxMother==probe_fj.genBhadron[0].genPartIdxMother):
                        probe_fj.genBhadron.append(gp)
                if gp.pdgId in [4, -4] and gp.genPartIdxMother>=0 and genparts[gp.genPartIdxMother].pdgId in [21, 25] and deltaR(gp, probe_fj)<=self._jetConeSize:
                    if len(probe_fj.genChadron)==0 or (len(probe_fj.genChadron)>0 and gp.genPartIdxMother==probe_fj.genChadron[0].genPartIdxMother):
                        probe_fj.genChadron.append(gp)
            probe_fj.genBhadron.sort(key=lambda x: x.pt, reverse=True)
            probe_fj.genChadron.sort(key=lambda x: x.pt, reverse=True)
            # null padding
            probe_fj.genBhadron += [_NullObject() for _ in range(2-len(probe_fj.genBhadron))]
            probe_fj.genChadron += [_NullObject() for _ in range(2-len(probe_fj.genChadron))]

            # last parton information
            for ifj in range(2 if self._channel == 'qcd' else 1):
                fj = event.fatjets[ifj]
                fj.npart, fj.nbpart, fj.ncpart, fj.ngpart, fj.part_sumpt, fj.bpart_sumpt, fj.cpart_sumpt, fj.gpart_sumpt = 0, 0, 0, 0, 0, 0, 0, 0
                fj.npart50, fj.nbpart50, fj.ncpart50, fj.ngpart50, fj.part50_sumpt, fj.bpart50_sumpt, fj.cpart50_sumpt, fj.gpart50_sumpt = 0, 0, 0, 0, 0, 0, 0, 0
                fj.nlpart, fj.nlpart50, fj.lpart_sumpt, fj.lpart50_sumpt = 0, 0, 0, 0
                for var in ['np', 'gp', 'cp', 'bp', 'lp']:
                    for pt in ['', '50']:
                        exec('fj.{var}{pt}={{"pt":[], "eta":[], "phi":[]}}'.format(var=var, pt=pt))
                for gp in genparts:
                    if gp.status>70 and gp.status<80 and (gp.statusFlags & (1 << 13)) and abs(gp.pdgId) in [1,2,3,4,5,6,21] and gp.pt>=5 and deltaR(gp, fj)<=self._jetConeSize:
                        fj.npart += 1; fj.part_sumpt += gp.pt
                        for var in ['pt', 'eta', 'phi']:
                            exec('fj.np["{var}"].append(gp.{var})'.format(var=var))
                        self.h_gp_npart_2d.Fill(gp.eta, gp.phi)
                        if gp.pdgId in [5, -5]:
                            fj.nbpart += 1; fj.bpart_sumpt += gp.pt
                            for var in ['pt', 'eta', 'phi']:
                                exec('fj.bp["{var}"].append(gp.{var})'.format(var=var))
                            self.h_gp_bpart_2d.Fill(gp.eta, gp.phi)
                        elif gp.pdgId in [4, -4]:
                            fj.ncpart += 1; fj.cpart_sumpt += gp.pt
                            for var in ['pt', 'eta', 'phi']:
                                exec('fj.cp["{var}"].append(gp.{var})'.format(var=var))
                            self.h_gp_cpart_2d.Fill(gp.eta, gp.phi)
                        elif gp.pdgId == 21:
                            fj.ngpart += 1; fj.gpart_sumpt += gp.pt
                            for var in ['pt', 'eta', 'phi']:
                                exec('fj.gp["{var}"].append(gp.{var})'.format(var=var))
                            self.h_gp_gpart_2d.Fill(gp.eta, gp.phi)
                        else:
                            fj.nlpart += 1; fj.lpart_sumpt += gp.pt
                            for var in ['pt', 'eta', 'phi']:
                                exec('fj.lp["{var}"].append(gp.{var})'.format(var=var))
                            self.h_gp_lpart_2d.Fill(gp.eta, gp.phi)

                        if gp.pt>=50:
                            fj.npart50 += 1; fj.part50_sumpt += gp.pt
                            for var in ['pt', 'eta', 'phi']:
                                exec('fj.np50["{var}"].append(gp.{var})'.format(var=var))
                            self.h_gp_npart50_2d.Fill(gp.eta, gp.phi)
                            if gp.pdgId in [5, -5]:
                                fj.nbpart50 += 1; fj.bpart50_sumpt += gp.pt
                                for var in ['pt', 'eta', 'phi']:
                                    exec('fj.bp50["{var}"].append(gp.{var})'.format(var=var))
                                self.h_gp_bpart50_2d.Fill(gp.eta, gp.phi)
                            elif gp.pdgId in [4, -4]:
                                fj.ncpart50 += 1; fj.cpart50_sumpt += gp.pt
                                for var in ['pt', 'eta', 'phi']:
                                    exec('fj.cp50["{var}"].append(gp.{var})'.format(var=var))
                                self.h_gp_cpart50_2d.Fill(gp.eta, gp.phi)
                            elif gp.pdgId == 21:
                                fj.ngpart50 += 1; fj.gpart50_sumpt += gp.pt
                                for var in ['pt', 'eta', 'phi']:
                                    exec('fj.gp50["{var}"].append(gp.{var})'.format(var=var))
                                self.h_gp_gpart50_2d.Fill(gp.eta, gp.phi)
                            else:
                                fj.nlpart50 += 1; fj.lpart50_sumpt += gp.pt
                                for var in ['pt', 'eta', 'phi']:
                                    exec('fj.lp50["{var}"].append(gp.{var})'.format(var=var))
                                self.h_gp_lpart50_2d.Fill(gp.eta, gp.phi)


    def evalTagger(self, event, jets):
        for j in jets:
            if self._opts['run_tagger']:
                outputs = [p.predict_with_cache(self.tagInfoMaker, event.idx, j.idx, j) for p in self.pnTaggers]
                outputs = ensemble(outputs, np.mean)
                j.pn_Xbb = outputs['probXbb']
                j.pn_Xcc = outputs['probXcc']
                j.pn_Xqq = outputs['probXqq']
                j.pn_QCD = convert_prob(outputs, None, prefix='prob')
            else:
                if self.hasParticleNetProb:
                    j.pn_Xbb = j.ParticleNetMD_probXbb
                    j.pn_Xcc = j.ParticleNetMD_probXcc
                    j.pn_Xqq = j.ParticleNetMD_probXqq
                    j.pn_QCD = convert_prob(j, None, prefix='ParticleNetMD_prob')
                else:
                    j.pn_Xbb = j.particleNetMD_Xbb
                    j.pn_Xcc = j.particleNetMD_Xcc
                    j.pn_Xqq = j.particleNetMD_Xqq
                    j.pn_QCD = j.particleNetMD_QCD
            j.pn_XbbVsQCD = convert_prob(j, ['Xbb'], ['QCD'], prefix='pn_')
            j.pn_XccVsQCD = convert_prob(j, ['Xcc'], ['QCD'], prefix='pn_')
            j.pn_XccOrXqqVsQCD = convert_prob(j, ['Xcc', 'Xqq'], ['QCD'], prefix='pn_')

    def evalMassRegression(self, event, jets):
        for j in jets:
            if self._opts['run_mass_regression']:
                outputs = [p.predict_with_cache(self.tagInfoMaker, event.idx, j.idx, j) for p in self.pnMassRegressions]
                j.regressed_mass = ensemble(outputs, np.median)['mass']
            else:
                try:
                    j.regressed_mass = j.particleNet_mass
                except RuntimeError:
                    j.regressed_mass = 0

    def _get_filler(self, obj):

        def filler(branch, value, default=0):
            self.out.fillBranch(branch, value if obj else default)

        return filler

    def fillFatJetInfo(self, event, fatjets):
        for idx in ([1, 2] if self._channel == 'qcd' else [1]):
            prefix = 'fj_%d_' % idx
            fj = fatjets[idx - 1]

            if not fj.is_qualified:
                # fill zeros if fatjet fails probe selection
                print(self.out._branches.keys())
                for b in self.out._branches.keys():
                    if b.startswith(prefix):
                        self.out.fillBranch(b, 0)
                continue

            if self._fill_sv:
                if self.isMC:
                    pass
                    for flavor in ['gp','bp','cp','lp']:
                        for var in ['pt','eta','phi']:
                            for ptsuf in ['', '50']:
                                print(prefix + "{flavor}{pt}_{var}".format(flavor=flavor, pt=ptsuf, var=var))
                                exec('print(fj.n{flavor}art{pt})'.format(flavor=flavor, pt=ptsuf, var=var))
                                exec('print(fj.{flavor}{pt}["{var}"])\n'.format(flavor=flavor, pt=ptsuf, var=var))
                                exec('temp_var_ini=fj.{flavor}{pt}["{var}"]'.format(flavor=flavor, pt=ptsuf, var=var))
                                temp_var = locals()['temp_var_ini']
                                print(temp_var)
                                self.out.fillBranch(prefix + "{flavor}{pt}_{var}".format(flavor=flavor, pt=ptsuf, var=var), temp_var)
                    # self.out.fillBranch(prefix + "npart", fj.npart)
                    # self.out.fillBranch(prefix + "nbpart", fj.nbpart)
                    # self.out.fillBranch(prefix + "ncpart", fj.ncpart)
                    # self.out.fillBranch(prefix + "ngpart", fj.ngpart)
                    # self.out.fillBranch(prefix + "part_sumpt", fj.part_sumpt)
                    # self.out.fillBranch(prefix + "bpart_sumpt", fj.bpart_sumpt)
                    # self.out.fillBranch(prefix + "cpart_sumpt", fj.cpart_sumpt)
                    # self.out.fillBranch(prefix + "gpart_sumpt", fj.gpart_sumpt)
                    # self.out.fillBranch(prefix + "npart50", fj.npart50)
                    # self.out.fillBranch(prefix + "nbpart50", fj.nbpart50)
                    # self.out.fillBranch(prefix + "ncpart50", fj.ncpart50)
                    # self.out.fillBranch(prefix + "ngpart50", fj.ngpart50)
                    # self.out.fillBranch(prefix + "part50_sumpt", fj.part50_sumpt)
                    # self.out.fillBranch(prefix + "bpart50_sumpt", fj.bpart50_sumpt)
                    # self.out.fillBranch(prefix + "cpart50_sumpt", fj.cpart50_sumpt)
                    # self.out.fillBranch(prefix + "gpart50_sumpt", fj.gpart50_sumpt)