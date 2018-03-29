#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 15:44:45 2018

@author: bbenneke
"""



import matplotlib.pyplot as plt
plt.ioff()
import matplotlib
matplotlib.interactive(False)
matplotlib.is_interactive()
from matplotlib import colors as mcolors
import math

import numpy as np
import multiprocessing
from exotep import utilities as ut
from exotep.utilities import savepickle, loadpickle

from exotep import constants
import exotep.radutil as rad 

from exotep.fitlightcurve import fitlightcurve, fitparam, visitTxt, findPara
import exotep

import exotep.plotTools

import os

import exotep.pyspectrum
import pdb
import batman
import itertools

from copy import deepcopy
import time

import pickle

from terra import tval, pipeline

#import customtransit as custom

jdref=2450000
ext='.pdf'

resultsDirec = '../exotep_results'

def fitK2(pla,lcs,guess,fitname, direc, walkersPerPara=6):
    
    #---Setup Fitting Model------
    fit = fitlightcurve(fitname,pla,filename=fitname,direc=direc,defaultValueForPara=guess,customtransit=False)
    priorUnc = dict()
    priorUnc['per']=1
    #---Fitting Parameters------
    fit.para.append(fitparam('RpRs',label='Radius Ratio',guess=guess['RpRs'],  low=np.sqrt(100e-6),  high=0.1, srange=guess['RpRs']*0.2))
    fit.para.append(fitparam('Tc',label='Transit Time',guess=guess['Tc'],  low=guess['Tc']-5.0/24,  high=guess['Tc']+5.0/24, srange=0.3*5.0/24))
    if len(lcs)>1:
        fit.para.append(fitparam('per',label='Period',guess=guess['per'],  low=guess['per']-2,  high=guess['per']+2, srange=0.3*priorUnc['per']))
    fit.para.append(fitparam('T14',label='T14',guess=guess['T14'],  low=0.5*guess['T14'],  high=2.0*guess['T14'], srange=0.05*guess['T14']))

    for i in range(len(lcs)):
        scatteri='ScatterPPM_'+str(i)
        guess[scatteri]=np.median(lcs[i].scatter)*1e6
        fit.para.append(fitparam(scatteri,label=scatteri,guess=guess[scatteri],  low=0,  high=100000, srange=guess[scatteri]/2))
    
    #---Add Light Curves and Instrument Model Parameters------
    for lc in lcs:
        fit.addLightCurve(lc)

 
    #---Prepare MCMC run -------
    fit.dispFittingPara()
    fit.initTransitModel()
    fit.runInitGuess()
    
    #---Run MCMC-------
    nwalkers = None
    if nwalkers is None:
        nwalkers = len(fit.para) * walkersPerPara
    
    start=time.time()
    nsteps=2000
    fit.emceefit(nsteps=nsteps,nwalkers=nwalkers,printIterations=True)
    print 'MCMC run time = ',time.time()-start,'sec'
            
    return fit

#%%    
def transit_radius_tdur(fits): 
    #clumsily extract the transit parameters I want to plot
    transitparamsAll = []
    tdur, tdur_min, tdur_max = [],[],[]
    radii, radii_min, radii_max = [],[],[]
    for fit in fits:
        fit.runModelwithBestFit()
        bestfit = fit.bestfit
        #radius_info = ((txt.splitlines())[8]).split()
        #tdur_info = ((txt.splitlines())[10]).split()
        radii.append(bestfit.ix['RpRs']['50'])
        radii_min.append(bestfit.ix['RpRs']['-1sigma'])
        radii_max.append(bestfit.ix['RpRs']['+1sigma'])
        tdur.append(bestfit.ix['T14']['50'])
        tdur_min.append(bestfit.ix['T14']['-1sigma'])
        tdur_max.append(bestfit.ix['T14']['+1sigma'])
        transitparamsAll.append(fit.bestfitRun['transitparams'])
    
    #-Fine Transit Model-----------------------------------------------
    bjdFine,transmodelFine,timeFine=[],[],[]
    time, flux, fluxerr = [], [], []
    for i,fit in enumerate(fits):
        transitparams=transitparamsAll[i]
        lc = fit.lcs[0]
        flux.append(lc.flux)
        fluxerr.append(lc.scatter)
        bjdFineNew = np.linspace(lc.bjd[0],lc.bjd[-1],100000)
        bjdFine.append(bjdFineNew)
        mTransitFine = batman.TransitModel(fit.bestfitRun['transitparams'][0], bjdFineNew,transittype = lc.lctype)
        transmodelFine.append(mTransitFine.light_curve(fit.bestfitRun['transitparams'][0]))
        timeFineNew = (bjdFineNew - bjdFineNew[0])*24
        time.append((lc.bjd - lc.bjd[0])*24)
        timeFine.append(timeFineNew)
    
    #check for consistency between all PAIRS of transits
    indices = range(len(fits))
    passed = True
    for pair in itertools.combinations(indices, r=2):
        if(radii[pair[0]]>radii[pair[1]]):
            sigma1 = abs(radii[pair[0]]-radii[pair[1]])/np.sqrt(radii_min[pair[0]]**2+radii_max[pair[1]]**2)
        else:
            sigma1 = abs(radii[pair[0]]-radii[pair[1]])/np.sqrt(radii_min[pair[1]]**2+radii_max[pair[0]]**2)
        if(tdur[pair[0]]>tdur[pair[1]]):
            sigma2 = abs(tdur[pair[0]]-tdur[pair[1]])/np.sqrt(tdur_min[pair[0]]**2+tdur_max[pair[1]]**2)
        else:
            sigma2 = abs(tdur[pair[0]]-tdur[pair[1]])/np.sqrt(tdur_min[pair[1]]**2+tdur_max[pair[0]]**2)
        if(sigma1>4 or sigma2>4):
            passed=False
        
    #make the plot to pass back
    depth = np.mean(radii)**2
    ystep=depth*2
    fig = plt.figure(figsize=(20,12))
    ax1 = plt.subplot2grid((2, 2), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((2, 2), (0, 1), rowspan=2)
    yshift=0
    colors = ['tomato','royalblue', 'mediumseagreen', 'gold', 'darkorchid', 'teal', 'darkorange', 'lightpink', 'cyan']
    for i in range(len(fits)):
        color = colors[i%len(colors)]
        ax1.plot(timeFine[i], transmodelFine[i]-yshift, color)
        ax1.errorbar(time[i],flux[i]-yshift,yerr=fluxerr[i],label='raw data',fmt='o',lw=1,ms=4,alpha=1, color=color)
        ax2.errorbar(tdur[i], radii[i], yerr=[[radii_min[i]], [radii_max[i]]], xerr=[[tdur_min[i]], [tdur_max[i]]], fmt='o', color=color)
        yshift+=ystep
    ax2.set_xlabel('T14 [seconds]')
    ax2.set_ylabel('Rp/Rstar')
    
    return passed, fig, ax1, ax2
#%%
def findNearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array-value)).argmin()
    return idx

def transit_residuals(fits):
    if(len(fits)>1):
        models, tdur, tc, scatter, transitparamsAll = [],[],[],[],[]
        for fit in fits:
            fit.runModelwithBestFit()
            scatter.append(fit.bestfitRun['scatter'])
            tdur.append(fit.bestfit.ix['T14']['50'])
            tc.append(fit.bestfit.ix['Tc']['50'])
            models.append(fit.bestfitRun['model'][0])
            transitparamsAll.append(fit.bestfitRun['transitparams'])
            
        #print (transitparamsAll[0][0]).rp, (transitparamsAll[0][0]).t0
        #print np.shape(models[1]), np.shape(fits[1].lcs[0].flux), np.shape(fits[1].lcs[0].bjd)
        
        #-Fine Transit Model-----------------------------------------------
        time, flux, fluxerr = [], [], []
        start, end, avg_in, avg_out_l, avg_out_r, t0 = [], [], [], [], [], []
        bjdFine,transmodelFine,timeFine=[],[],[]
        for i,fit in enumerate(fits):
            transitparams = transitparamsAll[i]
            lc = fit.lcs[0]
            flux.append(lc.flux)
            fluxerr.append(lc.scatter)
            timeNew = (lc.bjd - lc.bjd[0])*24
            time.append(timeNew)
            bjdFine.append(np.linspace(lc.bjd[0],lc.bjd[-1],100000))
            timeFine.append((bjdFine[i] - lc.bjd[0])*24)
            mTransitFine = batman.TransitModel(fit.transitparams0[0], bjdFine[i],transittype = lc.lctype)
            transmodelFine.append(mTransitFine.light_curve(transitparams[0]))
            t0.append(timeFine[i][transmodelFine[i].argmin()])
            #t0.append(transitparams[0].t0 + (i+1)*transitparams[0].per - lc.bjd[0])
            start.append(findNearest(time[i], math.floor(t0[i]-(tdur[i]/3600))))
            end.append(findNearest(time[i], math.ceil(t0[i]+(tdur[i]/3600))))
            avg_in.append(np.mean(abs(flux[i][start[i]:end[i]]-models[i][start[i]:end[i]])))
            avg_out_l.append(np.mean(abs(flux[i][:start[i]]-models[i][:start[i]])))
            avg_out_r.append(np.mean(abs(flux[i][end[i]:]-models[i][end[i]:])))
        
        ystep = 400
        #ystep = 0.01
        fig, ax = plt.subplots(figsize=(8,8))
        yshift=0
        passed=True
        colors = ['tomato','royalblue', 'mediumseagreen', 'gold', 'darkorchid', 'teal', 'darkorange', 'lightpink', 'cyan']
        for i in range(len(fits)):
            print avg_in[i], avg_out_l[i], avg_out_r[i]
            color = colors[i%len(colors)]
            if(avg_in[i]>(2.0*avg_out_l[i])) or (avg_in[i]>(2.0*avg_out_r[i])):
                passed = False
#            ax.plot(time[i], models[i]-yshift, color)
#            ax.errorbar(time[i][start[i]:end[i]],flux[i][start[i]:end[i]]-yshift,yerr=fluxerr[i][start[i]:end[i]],label='raw data',fmt='o',lw=1,ms=4,alpha=1, color=color)
#            ax.axvline(x=t0[i])
#            ax.axvline(x=t0[i])
            ax.errorbar(time[i][:start[i]],   (flux[i][:start[i]]-models[i][:start[i]])*1e6-yshift,  yerr=scatter[i][0]*np.ones(flux[i][:start[i]].shape)*1e6,fmt='o', lw=1, ms=4,label='',alpha=1, color=color)
            ax.errorbar(time[i][start[i]:end[i]],   (flux[i][start[i]:end[i]]-models[i][start[i]:end[i]])*1e6-yshift,  yerr=scatter[i][0]*np.ones(flux[i][start[i]:end[i]].shape)*1e6, fmt='o', lw=1, ms=4,alpha=1, color='k')
            ax.errorbar(time[i][end[i]:],   (flux[i][end[i]:]-models[i][end[i]:])*1e6-yshift,  yerr=scatter[i][0]*np.ones(flux[i][end[i]:].shape)*1e6, fmt='o', lw=1, ms=4,alpha=1, color=color)
            ax.axhline(y=-1*yshift, color='k')
            yshift+=ystep
    
    else:
        fit = fits[0]
        fit.runModelwithBestFit()
        scatter = fit.bestfitRun['scatter']
        tdur = fit.bestfit.ix['T14']['50']
        tc = fit.bestfit.ix['Tc']['50']
        models = fit.bestfitRun['model']
        
        #-Fine Transit Model-----------------------------------------------
        #bjdFine,transmodelFine,timeFine=[],[],[]
        time, flux = [], []
        start, end, avg_in, avg_out_l, avg_out_r, t0 = [], [], [], [], [], []
        for i,lc in enumerate(fit.lcs):
            flux.append(lc.flux)
            timeNew = (lc.bjd - lc.bjd[0])*24
            time.append(timeNew)
            t0.append(timeNew[models[i].argmin()])
            start.append(findNearest(timeNew, math.floor(t0[i]-(tdur/3600))))
            end.append(findNearest(timeNew, math.ceil(t0[i]+(tdur/3600))))
            avg_in.append(np.mean(abs(flux[i][start[i]:end[i]]-models[i][start[i]:end[i]])))
            avg_out_l.append(np.mean(abs(flux[i][:start[i]]-models[i][:start[i]])))
            avg_out_r.append(np.mean(abs(flux[i][end[i]:]-models[i][end[i]:])))
        
        ystep = 400
        fig, ax = plt.subplots(figsize=(8,8))
        yshift=0
        passed=True
        colors = ['tomato','royalblue', 'mediumseagreen', 'gold', 'darkorchid', 'teal', 'darkorange', 'lightpink', 'cyan']
        for i in range(len(fit.lcs)):
            color = colors[i%len(colors)]
            print avg_in[i], avg_out_l[i], avg_out_r[i]
            if(avg_in[i]>(2.0*avg_out_l[i])) or (avg_in[i]>(2.0*avg_out_r[i])):
                passed = False
            ax.errorbar(time[i][:start[i]],   (flux[i][:start[i]]-models[i][:start[i]])*1e6-yshift,  yerr=scatter[i]*np.ones(flux[i][:start[i]].shape)*1e6,fmt='o', lw=1, ms=4,label='',alpha=1, color=color)
            ax.errorbar(time[i][start[i]:end[i]],   (flux[i][start[i]:end[i]]-models[i][start[i]:end[i]])*1e6-yshift,  yerr=scatter[i]*np.ones(flux[i][start[i]:end[i]].shape)*1e6, fmt='o', lw=1, ms=4,alpha=1, color='k')
            ax.errorbar(time[i][end[i]:],   (flux[i][end[i]:]-models[i][end[i]:])*1e6-yshift,  yerr=scatter[i]*np.ones(flux[i][end[i]:].shape)*1e6, fmt='o', lw=1, ms=4,alpha=1, color=color)
            ax.axhline(y=-1*yshift, color='k')
            yshift+=ystep
    
    return passed, fig, ax


def showResults(fit,plotChain=True,plotTriangle=True):


    #---Postprocessing------
    fit.calcParaEstimates()
    fit.runModelwithBestFit()
    
    print fit.para, fit.chain.shape[1]

    #---Plot results-------
    if plotChain and len(fit.para)<=30 and fit.chain.shape[1]>150:
        fit.chainplot()   #Takes an extremely long time
    if plotTriangle and fit.chain.shape[1]>150:
        fit.triangleplot(onlyAstroPara=True)
    if plotTriangle and len(fit.para)<=30 and fit.chain.shape[1]>150:    
        fit.triangleplot(onlyAstroPara=False)

    #---Plot best fits    
    figs,axs=fit.plotLastLikelihoodEvaluation(plotOrigScatter=False,plotRawData=True,plotLCsInOneFigure=False,showDataIndex=True,figsize=[9,10])
    for i,lc in enumerate(fit.lcs):
        figs[i].savefig(fit.filebase + 'BestFit_LC'+'_.png')

    #---Useful outputs-------
    #fit.bestfit            #pandas dataframe with bestfit and uncertainties for each fitting parameter
    #fit.panda              #pandas dataframe with complete chains
    #fit.bestfitRun         #dictionary with all transit models and systematicsmodel for all light curves

    

if __name__ == "__main__":

    with open("../terra_analysis/pipe_pickle2",'r') as f:
        pipe = pickle.load(f)
    f.close()

    bjd = 249223471
    
    label_transit_kw = dict(cpad=0, cfrac=2)
    local_detrending_kw = dict(
        poly_degree=1, min_continuum=2, label_transit_kw=label_transit_kw
    )

    # Compute initial parameters. Fits are more robust if we star with
    # transits that are too wide as opposed to to narrow
    #for j in range(len(pipe.rows)):
    for j in range(1):
        j=1
        pipeline.choose_peak(pipe, j)
        P = pipe.grid_P
        t0 = pipe.grid_t0 
        tdur = pipe.grid_tdur
        rp = pipe.grid_rp
        b = 0.5
        
        #detrend the light curve w/ Terra
        lcdt = pipe.lc.copy()
        lcdt = lcdt[~lcdt.fmask].drop(['ftnd_t_roll_2D'],axis=1)
        lcdt = tval.local_detrending(lcdt, P, t0, 2.0 * tdur, **local_detrending_kw)
            
        lcs=[]
        i=0
        for transit_id, idx in lcdt.groupby('transit_id').groups.iteritems():
            transit = dict(transit_id=transit_id)
            lcdt_single = lcdt.ix[idx]
            t = np.array(lcdt_single.t) 
            f = np.array(lcdt_single.f)+1  
            lc= exotep.lightcurve.lightcurve('K2test_transit'+str(i),t,f)
            
            #I added the lc.visit, lc.lctype, lc.detrendPara
            lc.visit=str(i)
            lc.lctype = 'primary'
            lc.depthname = None
            lc.instrname='K2'
            lc.depthname='K2'
            lc.ldFit='fit'
            lc.ldlaw='linear'
            lc.ldCoef = np.array([0.2])
            lc.sysModel={'addParas': exotep.sysModels.addParas_LinWithDetrend, 'sysModel': exotep.sysModels.sysModel_LinWithDetrend, 'guess':None}
            lc.detrendPara = lc.bjd
            lcs.append(lc)
            i+=1
    
        
        guess=dict()
        guess['RpRs']=rp
        guess['Tc']= t0
        
        #extra things I had to initialize
        guess['Tsec']= t0
        guess['ecc']= 0
        guess['omega']= 0
        
        #the transit duration is in seconds but the period is not
        guess['T14']= tdur*24*3600
        guess['per']=P
        guess['b']=b
        guess['c']=1.0
        guess['v']=0.0
        #guess['ld0_K2']=
        
        pla=exotep.pyplanet.Planet()
        pla.Mstar=0.86*constants.Msun
        pla.Rstar=0.86*constants.Rsun #Crossfield
        pla.Teffstar=5261  #K
        pla.Rp=1.70*constants.Rearth
        pla.Mp=5.02*constants.Mearth
        pla.per=0.959628*constants.day #s
        pla.T14=guess['T14']*3600*24 #s
        
#        fig, ax = plt.subplots()
#        ax.scatter(lcs[0].bjd[25:-20], lcs[0].flux[25:-20])
#        plt.title('Peak 0, Transit 0')
#        plt.xlabel('Julian Date')
#        plt.ylabel('Flux')
#        plt.savefig('transit0.png')
        
        fits = []
        for i in range(len(lcs)):
            guess['Tc'] = t0+(P*i)
            matplotlib.interactive(False)
            direc = '../exotep_results/K2Fitting/Peak'+str(j)
            fitname = str(bjd)+'_peak'+str(j)+'_fit'+str(i)+'_'
            fit = fitK2(pla, [lcs[i]], guess, fitname, direc)
#            showResults(fit)
            fits.append(fit)
#        
#        with open("fits_Peak"+str(j),'r') as f:
#            fits = pickle.load(f)
#        f.close()
        with open("fits_Peak"+str(j),'wb') as f:
            pickle.dump(fits, f)
        f.close()
        
#        fit=fits[1]
#        fit.runModelwithBestFit()
#        model = fit.bestfitRun['model'][0]
#        print model
#        flux = fit.lcs[0].flux
#        d = fit.lcs[0].bjd
#        fig, ax = plt.subplots()
##        ax.plot(bjd, model)
#        ax.plot(d, flux)
#        
#        bjdFine=np.linspace(lc.bjd[0],lc.bjd[-1],100000)
#        print fit.transitparams0[0]
#        print fit.bestfitRun['transitparams'][0]
#        mTransitFine = batman.TransitModel(fit.bestfitRun['transitparams'][0], bjdFine, transittype = lc.lctype)
#        transmodelFine=mTransitFine.light_curve(fit.bestfitRun['transitparams'][0])
#        ax.plot(bjdFine, transmodelFine)
#        plt.show()
#        
        
##        
#        
        passed_1, fig_1, ax1_1, ax2_1 = transit_radius_tdur(fits)
        passed_2, fig_2, ax_2 = transit_residuals(fits)
        if passed_1 and passed_2:
            fig_1.savefig('../exotep_results/passed/' + str(bjd) + '_peak' + str(j) + '_radiusT14' + '.pdf')
            fig_2.savefig('../exotep_results/passed/' + str(bjd) + '_peak' + str(j) + '_residuals' + '.pdf')
        else:
            if passed_1:
                fig_1.savefig('../exotep_results/failed/' + str(bjd) + '_peak' + str(j) + '_radiusT14_PASS' + '.pdf')
            else:
                fig_1.savefig('../exotep_results/failed/' + str(bjd) + '_peak' + str(j) + '_radiusT14_FAIL' + '.pdf')
            if passed_2:
                fig_2.savefig('../exotep_results/failed/' + str(bjd) + '_peak' + str(j) + '_residuals_PASS' + '.pdf')
            else:
                fig_2.savefig('../exotep_results/failed/' + str(bjd) + '_peak' + str(j) + '_residuals_FAIL' + '.pdf')
        
#        passed_2, fig_2, ax_2 = transit_residuals(fits)
#        if passed_2:
#            plt.savefig('../exotep_results/passed/' + str(bjd) + '_peak_' + str(j) + '_residuals' + '.pdf')
#        else:
#            plt.savefig('../exotep_results/failed/' + str(bjd) + '_peak_' + str(j) + '_residuals' + '.pdf')
            
#        passed_3, fig_3, ax_3 = transit_residuals([jointfit])
#        if passed_3:
#            plt.savefig('../exotep_results/passed/' + str(bjd) + '_peak_' + str(j) + '_jointResiduals' + '.pdf')
#        else:
#            plt.savefig('../exotep_results/failed/' + str(bjd) + '_peak_' + str(j) + '_jointResiduals' + '.pdf')

#    jointfit = fitK2(pla,lcs,guess, 'jointfit')
#    jointfit.bestfit
#    showResults(jointfit)
    
#    fit0 = fitK2(pla,[lcs[0]],guess, 'fit0')
#    fit0.bestfit
#    showResults(fit0)
#    
#    fit1 = fitK2(pla,[lcs[1]],guess)
#    fit1.bestfit
#    showResults(fit)
    
#    fit.bestfit.ix['Dppm']['50']
#    fit.bestfit.ix['Dppm']['-1sigma']
#    fit.bestfit.ix['Dppm']['+1sigma']
#    fit.bestfit.ix['T14']['50']
#    
#    diff = fit1.bestfit.ix['Dppm']['50'] - fit0.bestfit.ix['Dppm']['50']


##%%
#
#    
#
#    #Script here   
#    
#    compDepth
#
#

        #jointfit = fitK2(pla,lcs,guess, 'jointfit')
        #ld_coef = jointfit.bestfit.ix['ld0_K2']['50']
        #showResults(jointfit, 'jointfit')
        #ld_coef=0.2


#fit0.__dict__
#fit0.__dict__.keys()











