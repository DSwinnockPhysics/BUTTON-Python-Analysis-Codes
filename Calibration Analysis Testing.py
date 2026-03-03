import uproot
import numpy
import matplotlib.pyplot as pyplot
import awkward as ak
import pandas as pd
pyplot.rcdefaults()
pyplot.rcParams["font.size"]=36
#pyplot.rcParams["figure.dpi"]=70

import scipy


def GaussianForFit(x, mean, sigma, amplitude):
    return amplitude * numpy.exp(-1 * ((x-mean)/(numpy.sqrt(2)*sigma))**2)


def LogNormalForFit(x, sigma, theta, m):
    exponent = numpy.log((x-theta)/m)**2 / (2*sigma**2) 
    return numpy.exp(-exponent) / ((x-theta)*sigma*numpy.sqrt(2*numpy.pi))



def DoubleGaussianForFit(x, mean, sigmaLeft, sigmaRight, amplitude):
    
    # A = (√2π(σ1 + σ2)/2)−1
    # Need to calculate amplitude such that the two sides are joined in the middle
    amplitudeMatch = ( numpy.sqrt(2*numpy.pi) * (sigmaLeft + sigmaRight) / 2 ) ** -1
    output = numpy.zeros(x.shape)
    
    leftMask = x < mean
    
    output[leftMask] = amplitude * amplitudeMatch * numpy.exp(-1 * ((x[leftMask]-mean)/(numpy.sqrt(2)*sigmaLeft))**2)
    
    output[~leftMask] = amplitude * amplitudeMatch * numpy.exp(-1 * ((x[~leftMask]-mean)/(numpy.sqrt(2)*sigmaRight))**2)
    
    #if x<mean:
    #    return amplitude * amplitudeMatch * numpy.exp(-1 * ((x-mean)/sigmaLeft)**2)
    #else:
    #    return amplitude * amplitudeMatch * numpy.exp(-1 * ((x-mean)/sigmaRight)**2)
    return output


def TwoGaussiansForFit(x, mean1, sigma1, amplitude1, mean2, sigma2, amplitude2):
    gaussian1 = amplitude1 * numpy.exp(-1 * ((x-mean1)/(numpy.sqrt(2)*sigma1))**2)
    gaussian2 = amplitude2 * numpy.exp(-1 * ((x-mean2)/(numpy.sqrt(2)*sigma2))**2)
    return gaussian1 + gaussian2




def TestDoubleGaussian():
    amplitude = 5
    xs = numpy.arange(-5,7, step=0.1)
    ys = DoubleGaussianForFit(xs, mean=1, sigmaLeft=1, sigmaRight=0.5, amplitude = amplitude)
    fig,ax = pyplot.subplots(1, 1, figsize=[32,16])
    ax.plot(xs, ys)
    
    # Check integration is close to amplitude * 1
    integration = numpy.trapz(ys,xs)
    print("Integration =",integration)
    print("Amplitude =", amplitude)

    return 0


def ExponentialForFit(x, alpha):
    output = numpy.zeros(x.shape)
    zeroMask = x <= 0
    
    output[~zeroMask] = 2000*numpy.exp(-alpha*x[~zeroMask])# alpha * numpy.exp(-alpha*x[~zeroMask])
    
    return output
    




def CombinedModelForFit(x,
                        doubleGaussianMean, doubleGaussianSigmaLeft, doubleGaussianSigmaRight, doubleGaussianAmplitude,
                        exponentialAlpha,
                        firstGaussianMean, firstGaussianSigma, firstGaussianAmplitude,
                        secondGaussianMean, secondGaussianSigma, secondGaussianAmplitude
                        ):
    # Double Gaussian fit to the pedestal
    # Exponential fit to the valley
    # Gaussian fits to the single + double peaks
    
    doubleGaussianPart = DoubleGaussianForFit(x, doubleGaussianMean, doubleGaussianSigmaLeft, doubleGaussianSigmaRight, doubleGaussianAmplitude)
    exponentialPart = ExponentialForFit(x, exponentialAlpha)
    firstGaussianPart = GaussianForFit(x, firstGaussianMean, firstGaussianSigma, firstGaussianAmplitude)
    secondGaussianPart = GaussianForFit(x, secondGaussianMean, secondGaussianSigma, secondGaussianAmplitude)
        
    
    #  This isn't actually right, there are some probabilistic elements for the exponential etc. that correspond to background stuff
    return doubleGaussianPart + exponentialPart + firstGaussianPart + secondGaussianPart



def TestCombinedModelForFit():
    xs = numpy.arange(-50, 1000, step = 0.1)
    
    doubleGaussianParams = [0, 5, 10, 2540000]
    exponentialParams = [0.012]
    firstGaussianParams = [190,50,6000]
    secondGaussianParams = [340, 60, 220]
    
    figCombinedTestPlot, axCombinedTestPlot = pyplot.subplots(1, 1, figsize=[32,16])
    axCombinedTestPlot.plot(xs, DoubleGaussianForFit(xs, *doubleGaussianParams), label="Double Gaussian", linestyle="--", color="r")
    axCombinedTestPlot.plot(xs, ExponentialForFit(xs, *exponentialParams), label="Exponential", linestyle="--", color="b")
    axCombinedTestPlot.plot(xs, GaussianForFit(xs, *firstGaussianParams), label="First Gaussian", linestyle="--", color="g")
    axCombinedTestPlot.plot(xs, GaussianForFit(xs, *secondGaussianParams), label="Second Gaussian", linestyle="--", color="orange")
    
    axCombinedTestPlot.plot(xs, CombinedModelForFit(xs, *doubleGaussianParams, *exponentialParams, *firstGaussianParams, *secondGaussianParams), label="Combined")
    
    axCombinedTestPlot.set_yscale("log")
    axCombinedTestPlot.set_ylim(100, 250000)
    axCombinedTestPlot.legend()
    axCombinedTestPlot.set_xlabel(r"Relative Charge (mV ns $\Omega$)")
    axCombinedTestPlot.set_ylabel("Entries")
    
    
    return 0



def PolyaForFit(x, mean, theta, invNBar):
    # # invVar = inverse of variance
    # # mean = gain
    # # n would be the incoming data I'd imagine (on the x axis)
    # # no idea what l is, or if it's something else like -1 or something, the writing is unclear and I can't find anything online about this form of the function
    # firstTerm = invVar**invVar / scipy.special.gamma(invVar)
    # print(firstTerm)
    # secondTerm = 1/mean * (n / mean)**(invVar*l)
    # print(secondTerm)
    # thirdTerm = numpy.exp(-invVar*n/mean)
    # print(thirdTerm)
    
    # Looking online it seems to be a scaled version of a gamma distribution rather than a Polya but maybe I've missed something
    # Either way, the version from Deb's code works this way:
    nbar = mean
    m = theta+1
    
    
    firstTerm = invNBar
    secondTerm = m**m/scipy.special.gamma(m)
    thirdTerm = (x/nbar)**(m-1)
    fourthTerm = numpy.exp(-m*x/nbar)
    
    
    return firstTerm*secondTerm*thirdTerm*fourthTerm



def TestPolyaForFit():
    xs = numpy.arange(0e-11, 4e-11, step=1e-13)*1e12
    #xs = numpy.arange(500, 50000, step=100)
    theta = 0.8e1 # This is gain fluctuation/variance
    mean = 1e-11*1e12
    invNBar=1e3 # For fitting a histogram, this is kind of like counts max (but not quite)
    ys = PolyaForFit(xs, mean, theta, invNBar)
    
    print("Ys:",ys)
    
    figPolyaTest, axPolyaTest = pyplot.subplots(1, 1, figsize=[16,8])
    axPolyaTest.plot(xs, ys)








def samplesToTime(sample):
    sampleRate = 500e6
    return sample/sampleRate

def timeToSamples(time):
    sampleRate = 500e6
    return time*sampleRate


# def GainFormula(singlePECharge):
#     # Not to be trusted, based on watchman paper 
#     electronCharge = 1.602176634e-19 # Coulombs
#     impedance = 50 # Ohms
#     ampGain = 1# 10 # 1 for ratpac sim, 10 in the paper they used
#     gain = (2 * singlePECharge * 10**-12) / (electronCharge * ampGain * impedance)
#     return gain









def CreateMinimaPulseData(numberOfPMTs, file, output, combinedPMTPulseData, digitiserProperties, **kwargs):
    
    # First gather the minima and store each in an array where each index corresponds to that pmt
    waveforms = file["waveforms"]["waveform"].array()
    pmtIDs = ak.flatten( output["digitPMTID"].array() )
    # print(pmtIDs)
    # print("Check lengths (pmtIds, waveforms):",len(pmtIDs),len(waveforms))
    
    
    #vHigh = digitiserProperties["vHigh"]# 1000 # mV
    vLow = digitiserProperties["vLow"]# -1000 # mV
    vOffset = digitiserProperties["vOffset"]# 800 # mV
    adcCountsPerMilliV =digitiserProperties["adcCountsPerMilliV"] # 16384 / (vHigh-vLow) 
    
                
    pmtMinima = numpy.empty(numberOfPMTs, dtype="object")
    for i in range(len(pmtMinima)):
        pmtMinima[i] = []

    
    for i in range(len(waveforms)):
        ####### MODIFIED TO REMOVE EXCESS DATA GENERATED FOR THE 500MS/s CASE, CHECK THIS IF THE SAMPLE RATE IS CHANGED
        currentMinimumSamples = numpy.amin(waveforms[i][:150])
        currentMinimumVolts =  ( currentMinimumSamples / adcCountsPerMilliV + vLow - vOffset) / 1000 
        currentPMTID = pmtIDs[i]
        pmtMinima[currentPMTID].append(currentMinimumVolts)
        combinedPMTPulseData[currentPMTID].append(currentMinimumVolts)

    checkHist = True
    if checkHist == True:
        
        numBins = kwargs.get('numBins', 20)
        diffuserName = kwargs.get('diffuserName',"Unknown Diffuser")
        
        
        
        # check a specific histogram:
        checkArrayIndex = 50
        
        figCheckMinimaHist = pyplot.figure(diffuserName + "_checkHist", figsize=[32,16])
        axCheckMinimaHist = figCheckMinimaHist.subplots(1, 1)
        minimumMinima = numpy.amin(pmtMinima[checkArrayIndex])
        maximumMinima = numpy.amax(pmtMinima[checkArrayIndex])
        minimaDifference = maximumMinima - minimumMinima
        
        binLowerBound = minimumMinima - minimaDifference*0.1
        binUpperBound = maximumMinima + minimaDifference*0.1
        binEdges = numpy.arange(binLowerBound, binUpperBound, step=minimaDifference/numBins)
        hist = axCheckMinimaHist.hist(pmtMinima[checkArrayIndex], bins=binEdges)
        histYs = hist[0]
        histXs = hist[1]
        #print(histXs)
        #print(histYs)
        axCheckMinimaHist.set_title(diffuserName + " Check Histogram ID " + str(checkArrayIndex))
        
        centeredXs = []
        for i in range(len(histXs)-1):
            centeredXs.append( (histXs[i]+histXs[i+1]) /2 )
        
        
        axCheckMinimaHist.set_xlabel("Peak Minimum Value (V)")
        axCheckMinimaHist.set_ylabel("Count")
        #axCheckMinimaHist.axvline(numpy.average(pmtMinima[longestArrayIndex]), color="orange")
    
        fitValues, fitCovariance = scipy.optimize.curve_fit( GaussianForFit, centeredXs, histYs , p0=[-0.014,0.002,12])
        #parameterError = numpy.sqrt(numpy.diag(fitCovariance))
        # print(parameterError)
        # print(fitCovariance)
        testXs = numpy.arange(histXs[0], histXs[-1], step=0.00001)
        testYs = GaussianForFit(testXs, *fitValues)
        axCheckMinimaHist.plot(testXs, testYs)
        axCheckMinimaHist.axvline(fitValues[0], color="orange") 
    return combinedPMTPulseData








def GetIntegrationValue(waveform, digitiserProperties):
    #vHigh = digitiserProperties["vHigh"]# 1000 # mV
    vLow = digitiserProperties["vLow"]# -1000 # mV
    vOffset = digitiserProperties["vOffset"]# 800 # mV
    adcCountsPerMilliV =digitiserProperties["adcCountsPerMilliV"] # 16384 / (vHigh-vLow) 
    sampleRate = digitiserProperties["sampleRate"]
    
    indexOfMinimum = numpy.argmin(waveform)
    width = 25 # samples
    minIntegrationIndex = indexOfMinimum - width
    if minIntegrationIndex < 0:
        minIntegrationIndex = 0
    maxIntegrationIndex = indexOfMinimum+width
    if maxIntegrationIndex > len(waveform):
        maxIntegrationIndex = len(waveform)
    firstTime = minIntegrationIndex / sampleRate
    lastTime = (maxIntegrationIndex+1) / sampleRate
    #times = numpy.arange(firstTime, lastTime, step = 1/sampleRate)
    
    #if maxIntegrationIndex < len(waveform)
    waveformToIntegrate = waveform[minIntegrationIndex:maxIntegrationIndex]
    times = numpy.linspace(firstTime, lastTime, num=len(waveformToIntegrate))
    waveformVolts = ( waveformToIntegrate / adcCountsPerMilliV + vLow - vOffset) / 1000 
    integration = numpy.trapz(waveformVolts, times)
    #print(integration)
    return integration








def CreateIntegrationPulseData(numberOfPMTs, file, output, combinedPMTPulseData, digitiserProperties, **kwargs):
    # First gather the minima and store each in an array where each index corresponds to that pmt
    waveforms = file["waveforms"]["waveform"].array()
    #print(waveforms)
    pmtIDs = ak.flatten( output["digitPMTID"].array() )
    # print(pmtIDs)
    # print("Check lengths (pmtIds, waveforms):",len(pmtIDs),len(waveforms))
    
    
    #vHigh = digitiserProperties["vHigh"]# 1000 # mV
    #vLow = digitiserProperties["vLow"]# -1000 # mV
    #vOffset = digitiserProperties["vOffset"]# 800 # mV
    #adcCountsPerMilliV = digitiserProperties["adcCountsPerMilliV"] # 16384 / (vHigh-vLow) 
    
                
    pmtIntegrations = numpy.empty(numberOfPMTs, dtype="object")
    for i in range(len(pmtIntegrations)):
        pmtIntegrations[i] = []

    
    for i in range(len(waveforms)):
        print(i)
        ####### MODIFIED TO REMOVE EXCESS DATA GENERATED FOR THE 500MS/s CASE, CHECK THIS IF THE SAMPLE RATE IS CHANGED
        currentIntegrationVolts = GetIntegrationValue(waveforms[i][:150], digitiserProperties)
        #print(currentIntegrationVolts)
        #currentIntegrationSamples = GetIntegrationValue(waveforms[i]) # numpy.amin(waveforms[i])
        #currentIntegrationVolts =  ( currentIntegrationSamples / adcCountsPerMilliV + vLow - vOffset) / 1000  # Units need establishing here because of the time integral
        currentPMTID = pmtIDs[i]
        pmtIntegrations[currentPMTID].append(currentIntegrationVolts)
        combinedPMTPulseData[currentPMTID].append(currentIntegrationVolts)

    checkHist = True
    if checkHist == True:
        
        numBins = kwargs.get('numBins', 20)
        diffuserName = kwargs.get('diffuserName',"Unknown Diffuser")
        
        
        
        # check a specific histogram:
        checkArrayIndex = 50
        
        figCheckIntegrationsHist = pyplot.figure(diffuserName + "_checkHist", figsize=[32,16])
        axCheckIntegrationsHist = figCheckIntegrationsHist.subplots(1, 1)
        minimumIntegrations = numpy.amin(pmtIntegrations[checkArrayIndex])
        maximumIntegrations = numpy.amax(pmtIntegrations[checkArrayIndex])
        IntegrationsDifference = maximumIntegrations - minimumIntegrations
        
        binLowerBound = minimumIntegrations - IntegrationsDifference*0.1
        binUpperBound = maximumIntegrations + IntegrationsDifference*0.1
        binEdges = numpy.arange(binLowerBound, binUpperBound, step=IntegrationsDifference/numBins)
        hist = axCheckIntegrationsHist.hist(pmtIntegrations[checkArrayIndex], bins=binEdges)
        histYs = hist[0]
        histXs = hist[1]
        #print(histXs)
        #print(histYs)
        axCheckIntegrationsHist.set_title(diffuserName + " Check Histogram ID " + str(checkArrayIndex))
        
        centeredXs = []
        for i in range(len(histXs)-1):
            centeredXs.append( (histXs[i]+histXs[i+1]) /2 )
        
        
        axCheckIntegrationsHist.set_xlabel("Integration Value (V ns)")
        axCheckIntegrationsHist.set_ylabel("Count")
        #axCheckIntegrationsHist.axvline(numpy.average(pmtIntegrations[longestArrayIndex]), color="orange")
        
        integrationFitParametersGuess = [-5e-11, 2e-11, 12]
        fitValues, fitCovariance = scipy.optimize.curve_fit( GaussianForFit, centeredXs, histYs , p0=integrationFitParametersGuess)
        #parameterError = numpy.sqrt(numpy.diag(fitCovariance))
        # print(parameterError)
        # print(fitCovariance)
        testXs = numpy.arange(histXs[0], histXs[-1], step=((histXs[-1]-histXs[0])/1000))
        testYs = GaussianForFit(testXs, *fitValues)
        axCheckIntegrationsHist.plot(testXs, testYs)
        axCheckIntegrationsHist.axvline(fitValues[0], color="orange") 
    
    return combinedPMTPulseData






# def SingleDiffuserAnalysis():
#     with uproot.open("150photons_DA40_output_1.ntuple.root") as file:
#         output = file["output"]
#         print("Check digitNhits:", output["digitNhits"].array())
#         print("Check digitPMTID:", output["digitPMTID"].array())
        
        
#         # First gather the minima and store each in an array where each index corresponds to that pmt
#         waveforms = file["waveforms"]["waveform"].array()
#         pmtIDs = ak.flatten( output["digitPMTID"].array() )
#         print(pmtIDs)
#         print("Check lengths (pmtIds, waveforms):",len(pmtIDs),len(waveforms))
        
#         numberOfPMTs = 96
        
        
#         pmtMinima = numpy.empty(numberOfPMTs, dtype="object")
#         for i in range(len(pmtMinima)):
#             pmtMinima[i] = []
        
        
        
#         vHigh = 1000 # mV
#         vLow = -1000 # mV
#         vOffset = 800 # mV   <- This is just a hack until I fix what the actual voltage offset should be in the simulation
#         adcCountsPerMilliV = 16384 / (vHigh-vLow) 
        
#         for i in range(len(waveforms)):
#             currentMinimumSamples = numpy.amin(waveforms[i])
#             currentMinimumVolts =  ( currentMinimumSamples / adcCountsPerMilliV + vLow - vOffset) / 1000 
#             currentPMTID = pmtIDs[i]
#             pmtMinima[currentPMTID].append(currentMinimumVolts)
            
        
        
        
#         # Test on the longest array, plot a histogram or something of the minima
#         longestArrayIndex = 0
#         longestArrayLength = 0
        
#         for i in range(len(pmtMinima)):
#             #print(pmtMinima[i])
            
#             pmtMinimaLength = len(pmtMinima[i])
            
#             if i == 95:
#                 longestArrayLength = pmtMinimaLength
#             else:
#                 if pmtMinimaLength > longestArrayLength:
#                     longestArrayIndex = i
#                     longestArrayLength = pmtMinimaLength
        
        
#         # check a specific histogram:
#         longestArrayIndex = 50
        
#         # figCheckMinimaHist, axCheckMinimaHist = pyplot.subplots(1, 1, figsize=[32,16])
#         figCheckMinimaHist = pyplot.figure(1, figsize=[32,16])
#         axCheckMinimaHist = figCheckMinimaHist.subplots(1, 1)
#         minimumMinima = numpy.amin(pmtMinima[longestArrayIndex])
#         maximumMinima = numpy.amax(pmtMinima[longestArrayIndex])
#         minimaDifference = maximumMinima - minimumMinima
        
#         binLowerBound = minimumMinima-minimaDifference*0.1
#         binUpperBound = maximumMinima+minimaDifference*0.1
#         binEdges = numpy.arange(binLowerBound, binUpperBound, step=minimaDifference/20)
#         hist = axCheckMinimaHist.hist(pmtMinima[longestArrayIndex], bins=binEdges)
#         histYs = hist[0]
#         histXs = hist[1]
#         #print(histXs)
#         #print(histYs)
        
#         centeredXs = []
#         for i in range(len(histXs)-1):
#             centeredXs.append( (histXs[i]+histXs[i+1]) /2 )
        
        
#         axCheckMinimaHist.set_xlabel("Peak Minimum Value (V)")
#         axCheckMinimaHist.set_ylabel("Count")
#         #axCheckMinimaHist.axvline(numpy.average(pmtMinima[longestArrayIndex]), color="orange")
        
        
                    
#         fitValues, fitCovariance = scipy.optimize.curve_fit( GaussianForFit, centeredXs, histYs , p0=[-0.014,0.002,12])
#         parameterError = numpy.sqrt(numpy.diag(fitCovariance))
#         print(parameterError)
#         print(fitCovariance)
#         testXs = numpy.arange(histXs[0], histXs[-1], step=0.00001)
#         testYs = GaussianForFit(testXs, *fitValues)
#         axCheckMinimaHist.plot(testXs, testYs)
#         #for i in range(len(centeredXs)):
#         #    axCheckMinimaHist.axvline(centeredXs[i])
#         axCheckMinimaHist.axvline(fitValues[0], color="orange") 
        
    
    
#         # need to make a lognormal to fit
#         # First test the lognormal:
#         #xs = numpy.arange(0.01,5, step=0.1)
#         xs = numpy.arange(-100.1e-9, 300.1e-9, step = 1e-9)
#         print(xs.shape)
#         ys = []
#         xList = []
#         for x in xs:
#             #print(x)
#             xList.append(x)
#             ys.append( LogNormalForFit(x, sigma=0.15e-3, theta=-10.5e-3, m=10.5e-3) )
#         print(len(ys))
        
#         figLognormalTest = pyplot.figure(1000, figsize=[32,16])
#         axLognormalTest = figLognormalTest.subplots(1, 1)
#         axLognormalTest.plot(xList, ys)
#         print(xList)
#         print(ys)
#         #axLognormalTest.set_yscale("log")
#         #axLognormalTest.set_ylim(10**-3,2)
#         #axLognormalTest.set_xlim(0,5)
#         axLognormalTest.set_xlabel("Time (s)")
        
#         # # Lognormal looks good, so now to fit
#         # # wait no
#         # # that's for the pulse shapes, which for now I'm using the minimum for which should be fine
#         # # If anything I want a better shape for the distribution of pulse heights (which is what?)
#         # It's gaussian-ish, so things should be fine (at least for 1 photoelectron, with coincidences it gets distorted further)
        
#         # Store the parameters for each pmtID
#         fittedParameters = numpy.empty(numberOfPMTs, dtype="object")
        
        
#         minPointsToFit = 20
#         numBins = 20
#         for i in range(len(pmtMinima)):
#             if len(pmtMinima[i]) >= minPointsToFit: # want a baseline number of pmt minima to actually fit to
#                 # print(i)
#                 minimumMinima = numpy.amin(pmtMinima[i])
#                 maximumMinima = numpy.amax(pmtMinima[i])
#                 minimaDifference = maximumMinima - minimumMinima
                
#                 binLowerBound = minimumMinima-minimaDifference*0.1
#                 binUpperBound = maximumMinima+minimaDifference*0.1
#                 binEdges = numpy.arange(binLowerBound, binUpperBound, step=minimaDifference/numBins)
                
#                 # Possibly need to create a separate figure to store the hists in else it affects previous plots
#                 histFigure = pyplot.figure(3)
#                 histAx = histFigure.subplots(1,1)
#                 currentHist = histAx.hist(pmtMinima[i], bins=binEdges)
#                 #currentHist = pyplot.hist(pmtMinima[i], bins=binEdges)
#                 currentHistYs = currentHist[0]
#                 currentHistXs = currentHist[1]
#                 centeredXs = []
#                 for j in range(len(currentHistXs)-1):
#                     centeredXs.append( (currentHistXs[j]+currentHistXs[j+1]) /2 )
                
#                 currentFitValues, currentFitCovariance = scipy.optimize.curve_fit( GaussianForFit, centeredXs, currentHistYs , p0=[-0.014,0.0025,12])
#                 fittedParameters[i] = currentFitValues
#                 # print(fittedParameters[i], fittedParameters[i-1])
        
#         pmtIDList = range(numberOfPMTs)
#         #fittedParameters = numpy.array(fittedParameters)
        
#         #for i in range(len(pmtMinima)):
#         #    print(len(pmtMinima[i]))
        
#         # print(len(fittedParameters))
#         # print(fittedParameters)
        
#         # for i in range(len(fittedParameters)):
#             # print(fittedParameters[i])
        
        
#         pmtFittedMeans = []
#         pmtFittedSigmas = []
#         for i in range(len(fittedParameters)):
#             #if fittedParameters[i].empty():
#             #    pmtFittedMeans = 0
#             #else:
#             if type(fittedParameters[i]) != type(None):
#                 pmtFittedMeans.append(fittedParameters[i][0])
#                 pmtFittedSigmas.append(abs( fittedParameters[i][1] )) # There was one with a negative sigma for some reason, and it seems to just be mirrored - Negative sigma on a gaussian is squared anyway 
#             else:
#                 pmtFittedMeans.append(0)
#                 pmtFittedSigmas.append(0)
           
#         #print(pmtFittedSigmas)
            
#         figFitAccuracy, axFitAccuracy = pyplot.subplots(1, 1, figsize=[32,16])
#         axFitAccuracy.scatter(pmtIDList, pmtFittedMeans)
#         axFitAccuracy.errorbar(pmtIDList, pmtFittedMeans, pmtFittedSigmas)
#         axFitAccuracy.set_xlabel("PMT ID")
#         axFitAccuracy.set_ylabel("1pe Response (V)")
        
        
        
#         # Now would want a plot of each one with the true values as well and some standard deviation/standard error stuff
        
        
#         trueMeanHeight = -12.65 /1000
#         trueHeightSpread = 6.522136474 /1000 / numpy.sqrt(2)
#         axFitAccuracy.scatter(96, trueMeanHeight, color="orange")
#         axFitAccuracy.errorbar(96, trueMeanHeight, trueHeightSpread, color="orange")
        
        
        
#         # Now want to try and quantify the accuracy of the thing with respect to the number of photons in
#         # First get the number of events used for each and then plot that against some measure of maybe mean first?
#         numberOfEventsInEach = []
#         meanDifferences = []
    
#         for i in range(len(pmtMinima)):
#             numberOfEventsInEach.append(len(pmtMinima[i]))
#             #print(pmtFittedMeans[i])
#             meanDifferences.append(pmtFittedMeans[i] - trueMeanHeight)
#             #print(meanDifferences[i])
            
#         numberOfEventsInEach = numpy.array(numberOfEventsInEach)
#         meanDifferences = numpy.array(meanDifferences)
#         percentageMeanDifferences = meanDifferences / trueMeanHeight
#         #print(numberOfEventsInEach.shape)
#         #print(meanDifferences.shape)
#         #print(len(numberOfEventsInEach))
#         #print(len(meanDifferences))
#         #print(numberOfEventsInEach)
#         #print(meanDifferences)
#         figMeanAccVsPhotonCount = pyplot.figure(4, figsize=[32,16])
#         axMeanAccVsPhotonCount = figMeanAccVsPhotonCount.subplots(1, 1)
#         axMeanAccVsPhotonCount.scatter(numberOfEventsInEach, percentageMeanDifferences)
#         axMeanAccVsPhotonCount.set_xlabel("Number of Hits")
#         axMeanAccVsPhotonCount.set_ylabel("Fractional Difference from True Mean")
#         #axMeanAccVsPhotonCount.set_ylim(-0.0035, 0.0035)
#         axMeanAccVsPhotonCount.set_ylim(-0.35, 0.35)
#         axMeanAccVsPhotonCount.set_xlim(0,5000)
#         axMeanAccVsPhotonCount.axhline(0)
#     return 0
    
    
    
    
def GetDigitiserProperties():
    vHigh = 1000 # mV
    vLow = -1000 # mV
    vOffset = 800 # mV
    adcCountsPerMilliV = 16384 / (vHigh-vLow) 
    sampleRate = 500e6 # 500MS/s - Although, this should match the data, is this stored somewhere in the output?  The meta branch perhaps?
    digitiserProperties = {"vHigh":vHigh,
                          "vLow":vLow,
                           "vOffset":vOffset,
                           "adcCountsPerMilliV":adcCountsPerMilliV,
                           "sampleRate":sampleRate
                           }
    return digitiserProperties
    



def GeneratePMTPulseDataFromFiles(fileNames, numberOfPMTs, sourceLabels, analysisType, numBins):
    fileIndex=0
    combinedPMTPulseData = numpy.empty(numberOfPMTs, dtype="object")
    for i in range(len(combinedPMTPulseData)):
        combinedPMTPulseData[i] = []
    
    digitiserProperties = GetDigitiserProperties()
    adcCountsPerMilliV = digitiserProperties["adcCountsPerMilliV"]
    vLow = digitiserProperties["vLow"]
    vOffset = digitiserProperties["vOffset"]
    
    for fileName in fileNames:
        print(fileIndex)
        
        sourceName = sourceLabels[fileIndex]
        with uproot.open(fileName) as file:
            output = file["output"]
            print("mcx length:",len(output["mcx"].array()))  

            figExampleWaveform, axExampleWaveform = pyplot.subplots(1, 1, figsize=[16,8])
            axExampleWaveform.plot((file["waveforms"]["waveform"].array()[0][:150] / adcCountsPerMilliV + vLow - vOffset) / 1000)

            if analysisType == "minima":
                combinedPMTPulseData = CreateMinimaPulseData(numberOfPMTs, file, output, combinedPMTPulseData, digitiserProperties, numBins = numBins, diffuserName=sourceName)
            elif analysisType == "integration":
                combinedPMTPulseData = CreateIntegrationPulseData(numberOfPMTs, file, output, combinedPMTPulseData, digitiserProperties, numBins = numBins, diffuserName=sourceName)
            else:
                raise Exception("Invalid analysis type, use \"integration\" or \"minima\".")
            
        fileIndex+=1    
    
    return combinedPMTPulseData




    
# Now need to do the fitting for 4 diffusers, kind of want to separate things out so I can just reuse stuff from the previous section but not have it all in one go every time
def GetMultiDiffuserPMTPulseData(numberOfPMTs, numBins, analysisType):
    """Reads the data for each diffuser and returns the minimum/integrated values of the voltage for each event.
    The returned array has a list for each pmt ID containing the minimum/integrated voltage values."""
    
    diffuserLabels = ["DA34", "DA37", "DA38", "DA40"]
    fileNames =[]
    for diffuserName in diffuserLabels:
        fileNames.append("150photons_buttondaq_" + diffuserName + "_30kEvents_LowSampleRate.ntuple.root")
    
    combinedPMTPulseData = GeneratePMTPulseDataFromFiles(fileNames, numberOfPMTs, diffuserLabels, analysisType, numBins)
    
    return combinedPMTPulseData



def GetLightBallPMTPulseData(numberOfPMTs, numBins, analysisType):
    fileName = ["CoincidenceStudy/10kEvents/coincidenceTest800Photons_lb.ntuple.root"]
    fileLabels = ["Light Ball"]
    combinedPMTPulseData = GeneratePMTPulseDataFromFiles(fileName, numberOfPMTs, fileLabels, analysisType, numBins)
    
    return combinedPMTPulseData
    








def FitMinima(minimaToFit, numBins):
    if len(minimaToFit) >= 20: # want a baseline number of pmt minima to actually fit to
        minimumMinima = numpy.amin(minimaToFit)
        maximumMinima = numpy.amax(minimaToFit)
        minimaDifference = maximumMinima - minimumMinima
        
        binLowerBound = minimumMinima-minimaDifference*0.1
        binUpperBound = maximumMinima+minimaDifference*0.1
        binEdges = numpy.arange(binLowerBound, binUpperBound, step=minimaDifference/numBins)
        
        # Possibly need to create a separate figure to store the hists in else it affects previous plots
        #histFigure = pyplot.figure(3)
        #histAx = histFigure.subplots(1,1)
        # currentHist = histAx.hist(combinedPMTPulseData[i], bins=binEdges)
        currentHist = numpy.histogram(minimaToFit, bins=binEdges)
        #currentHist = pyplot.hist(combinedPMTPulseData[i], bins=binEdges)
        currentHistYs = currentHist[0]
        currentHistXs = currentHist[1]
        centeredXs = []
        for j in range(len(currentHistXs)-1):
            centeredXs.append( (currentHistXs[j]+currentHistXs[j+1]) /2 )
        
        currentFitValues, currentFitCovariance = scipy.optimize.curve_fit( GaussianForFit, centeredXs, currentHistYs , p0=[-0.014,0.002,12])
        # fittedParameters[i] = currentFitValues
        # print(fittedParameters[i], fittedParameters[i-1])
        
        printTestHistogram = False
        if printTestHistogram == True:
            figTesthist, axTestHist = pyplot.subplots(1, 1, figsize=[32,16])
            testHist = axTestHist.hist(minimaToFit, bins=binEdges)
            axTestHist.set_ylabel("Count")
            axTestHist.set_xlabel("Peak Minimum Value (V)")
            testXs = numpy.arange(currentHistXs[0], currentHistXs[-1], step=0.00001)
            testYs = GaussianForFit(testXs, *currentFitValues)
            axTestHist.plot(testXs, testYs)
            axTestHist.axvline(currentFitValues[0], color="orange") 
    return currentFitValues


def FitGaussianIntegration(integrationsToFit, binEdges, currentHistXs, currentHistYs, centeredXs, centeredXsScaled, printTestHistogram):
    integrationFitParametersGuess = [-5e-11, 2e-11, 12]
    # Want to change these initial guesses into something based on the waveform for better starting parameters
    # Would be good to know the maximum value of the histogram and the std dev or something like that
    histogramMax = max(currentHistYs)
    # https://stackoverflow.com/questions/50786699/how-to-calculate-the-standard-deviation-from-a-histogram-python-matplotlib
    # First get the mean of the histogram with weighted average:
    midpointsOfBins = centeredXs
    histogramMean = numpy.average(midpointsOfBins, weights=currentHistYs)
    # "The estimated variance is the weighted average of the squared difference from the mean:"
    histogramVariance = numpy.average((midpointsOfBins - histogramMean)**2, weights=currentHistYs)
    histogramStdDev = numpy.sqrt(histogramVariance)
    print("Histogram mean:",histogramMean)
    print("Histogram Max:",histogramMax)
    print("Histogram StdDev:",histogramStdDev)
    
    # Use these parameters to refine the initial guess parameters
    integrationFitParametersGuess = [histogramMean, histogramStdDev, histogramMax]
    
    if printTestHistogram == True:
        testXs = numpy.arange(currentHistXs[0], currentHistXs[-1], step=(currentHistXs[1]-currentHistXs[0])/100)
        figInitial, axInitial = pyplot.subplots(1, 1, figsize=[32,16])
        axInitial.hist(integrationsToFit, bins=binEdges)
        axInitial.plot(testXs, GaussianForFit(testXs, *integrationFitParametersGuess), color="r")
        
    
    scaledIntegrationFitParametersGuess = [integrationFitParametersGuess[0]*10**12,
                                           integrationFitParametersGuess[1]*10**12,
                                           integrationFitParametersGuess[2]
                                           ]
    
    currentFitValues, currentFitCovariance = scipy.optimize.curve_fit(GaussianForFit,
                                                                      centeredXsScaled, 
                                                                      currentHistYs, 
                                                                      #p0=integrationFitParametersGuess
                                                                      p0=scaledIntegrationFitParametersGuess
                                                                      )
    # Un-scale the fit values 
    currentFitValues = [currentFitValues[0]*10**-12,
                        currentFitValues[1]*10**-12,
                        currentFitValues[2]
                        ]
        
    return currentFitValues, currentFitCovariance
            




def FitTwoGaussianIntegration(integrationsToFit, binEdges, currentHistXs, currentHistYs, centeredXs, centeredXsScaled, printTestHistogram):
    firstGaussianGuessParameters = [-5e-11, 2e-11, 100]
    secondGaussianGuessParameters = [-1.2e-10, 3e-11, 10]
    
    # Want to change these initial guesses into something based on the waveform for better starting parameters
    # Would be good to know the maximum value of the histogram and the std dev or something like that
    
    histogramMax = max(currentHistYs)
    # https://stackoverflow.com/questions/50786699/how-to-calculate-the-standard-deviation-from-a-histogram-python-matplotlib
    # First get the mean of the histogram with weighted average:
    midpointsOfBins = centeredXs
    histogramMean = numpy.average(midpointsOfBins, weights=currentHistYs)
    # "The estimated variance is the weighted average of the squared difference from the mean:"
    histogramVariance = numpy.average((midpointsOfBins - histogramMean)**2, weights=currentHistYs)
    histogramStdDev = numpy.sqrt(histogramVariance)
    
    print("Histogram Max:",histogramMax)
    print("Histogram StdDev:",histogramStdDev)
    
    # Use these parameters to refine the initial guess parameters
    firstGaussianGuessParameters = [histogramMean*0.8, 0.8*histogramStdDev, histogramMax]
    secondGaussianGuessParameters = [1.6*histogramMean, 0.8*histogramStdDev, histogramMax/3]
    
    
    
    combinedGuessParameters = firstGaussianGuessParameters + secondGaussianGuessParameters
    if printTestHistogram == True:
        testXs = numpy.arange(currentHistXs[0], currentHistXs[-1], step=(currentHistXs[1]-currentHistXs[0])/100)
        figInitial, axInitial = pyplot.subplots(1, 1, figsize=[32,16])
        axInitial.hist(integrationsToFit, bins=binEdges)
        axInitial.plot(testXs, TwoGaussiansForFit(testXs, *firstGaussianGuessParameters, *secondGaussianGuessParameters))
        axInitial.plot(testXs, GaussianForFit(testXs, *firstGaussianGuessParameters), color="r")
        axInitial.plot(testXs, GaussianForFit(testXs, *secondGaussianGuessParameters), color="b")
    
    # The parameters should be scaled such that they are all roughly the same order of magnitude, otherwise one variable can dominate the regression 
    

    scaledGuessParameters = [combinedGuessParameters[0]*10**12,
                             combinedGuessParameters[1]*10**12,
                             combinedGuessParameters[2],
                             combinedGuessParameters[3]*10**12,
                             combinedGuessParameters[4]*10**12,
                             combinedGuessParameters[5]
                             ]
    # m1, s1, a1, m2, s2, a2
    lowerBounds = [-numpy.inf, 0, 0, -numpy.inf, 0, 0]
    #lowerBounds = [-numpy.inf, -numpy.inf, -numpy.inf, -numpy.inf, -numpy.inf, -numpy.inf]
    upperBounds = [numpy.inf, numpy.inf, numpy.inf, numpy.inf, numpy.inf, numpy.inf]
    currentFitValues, currentFitCovariance = scipy.optimize.curve_fit(TwoGaussiansForFit,
                                                                      centeredXsScaled,
                                                                      currentHistYs,
                                                                      p0=scaledGuessParameters, 
                                                                      maxfev=8000,
                                                                      bounds = (lowerBounds,upperBounds)
                                                                      )



    # Un-scale the fit values 
    currentFitValues = [currentFitValues[0]*10**-12,
                        currentFitValues[1]*10**-12,
                        currentFitValues[2],
                        currentFitValues[3]*10**-12,
                        currentFitValues[4]*10**-12,
                        currentFitValues[5]
                        ]
    return currentFitValues, currentFitCovariance
        




def FitPolyaIntegration(integrationsToFit, binEdges, currentHistXs, currentHistYs, centeredXs, centeredXsScaled, printTestHistogram, fitType):        
    histogramMax = max(currentHistYs)
    # https://stackoverflow.com/questions/50786699/how-to-calculate-the-standard-deviation-from-a-histogram-python-matplotlib
    # First get the mean of the histogram with weighted average:
    midpointsOfBins = centeredXs
    histogramMean = numpy.average(midpointsOfBins, weights=currentHistYs)
    # "The estimated variance is the weighted average of the squared difference from the mean:"
    histogramVariance = numpy.average((midpointsOfBins - histogramMean)**2, weights=currentHistYs)
    histogramStdDev = numpy.sqrt(histogramVariance)
    
    print("Histogram Max:",histogramMax)
    print("Histogram StdDev:",histogramStdDev)
    
    # Use these parameters to refine the initial guess parameters
    
    integrationFitParametersGuess = [histogramMean, 6, histogramMax]
    
    print(integrationFitParametersGuess)
    
    if printTestHistogram == True:
        testXs = numpy.arange(currentHistXs[0], currentHistXs[-1], step=(currentHistXs[1]-currentHistXs[0])/100)
        figInitial, axInitial = pyplot.subplots(1, 1, figsize=[32,16])
        axInitial.hist(integrationsToFit, bins=binEdges)
        axInitial.plot(testXs, PolyaForFit(testXs, *integrationFitParametersGuess), color="r")
        
    # Want to scale the x axis for more accurate fitting, since curve_fit wants the parameters to be not too different in terms of order of magnitude
    scaledIntegrationFitParametersGuess = [integrationFitParametersGuess[0]*10**12,
                                           integrationFitParametersGuess[1],
                                           integrationFitParametersGuess[2]
                                           ]
    
    print(scaledIntegrationFitParametersGuess)
    print(centeredXsScaled)
    currentFitValues, currentFitCovariance = scipy.optimize.curve_fit(PolyaForFit,
                                                                      centeredXsScaled, 
                                                                      currentHistYs, 
                                                                      #p0=integrationFitParametersGuess
                                                                      p0=scaledIntegrationFitParametersGuess
                                                                      )
    # Un-scale the fit values 
    currentFitValues = [currentFitValues[0]*10**-12,
                        currentFitValues[1],
                        currentFitValues[2]
                        ]

    if fitType=="PolyaPeak":
        # Need to get the peak position and return that instead of the mean
        polyaXs = numpy.arange(min(centeredXs), max(centeredXs), step = (max(centeredXs)-min(centeredXs))/1000 )
        polyaYs = PolyaForFit(polyaXs, *currentFitValues)
        integrationValueAtPeak = polyaXs[numpy.argmax(polyaYs)]
        #print("Polya peak pos:",integrationValueAtPeak)
        polyaMean = currentFitValues[0] # store mean for later use in plotting
        currentFitValues[0] = integrationValueAtPeak # overwrite the fit values to return for analysis of the correct values
        # This is a bit of a mess and probably wants refactoring, but it's not entirely clear how to handle all these different cases
    else:
        polyaMean=0
    


    return currentFitValues, currentFitCovariance, polyaMean




def FitAdaptiveIntegration(integrationsToFit, binEdges, currentHistXs, currentHistYs, centeredXs, centeredXsScaled, printTestHistogram):
    # Uses one method, then fits with another if that one fails to fit based on the covariance
    # Start with single gaussian
    # I can use the preexisting single Gaussian code to make this simple maybe
    firstFitValues, firstFitCovariance = FitGaussianIntegration(integrationsToFit, binEdges, currentHistXs, currentHistYs, centeredXs, centeredXsScaled, printTestHistogram)
    
    fitCovarianceThreshold = 25
    perr = numpy.sqrt(numpy.diag(firstFitCovariance))
        # The covariance varies for the different parameters in the estimation
        # For a single gaussian I will have 3 parameters
        # Generally I only really care about the first gaussian anyway
        # There are 3 parameters then that I have to use to judge whether or not to try a two-gaussian fit
        # mean, sigma and amplitude
        # My initial thought is to keep things easy and just combine them all in quadrature
        # And then that way if even one of them is way off then it will sound the alarm
    #print(perr)
    combinedCovariance = numpy.sqrt( (perr[0])**2 + (perr[1])**2 + (perr[2])**2 )
    print("First fit covariance:",combinedCovariance)
    if combinedCovariance > fitCovarianceThreshold:
        print("First fit failed, trying second method.")
        secondFitValues, secondFitCovariance = FitTwoGaussianIntegration(integrationsToFit, binEdges, currentHistXs, currentHistYs, centeredXs, centeredXsScaled, printTestHistogram)
        return secondFitValues, secondFitCovariance
    else:
        return firstFitValues, firstFitCovariance
        



def printTestHistogramPlot(integrationsToFit, binEdges, currentHistXs, currentFitValues, fitType, polyaMean=0):
    figTesthist, axTestHist = pyplot.subplots(1, 1, figsize=[32,16])
    testHist = axTestHist.hist(integrationsToFit, bins=binEdges)
    axTestHist.set_ylabel("Count")
    axTestHist.set_xlabel("Integration Value (Vs)")
    testXs = numpy.arange(currentHistXs[0], currentHistXs[-1], step=(currentHistXs[1]-currentHistXs[0])/100)
    if fitType == "gaussian":
        testYs = GaussianForFit(testXs, *currentFitValues)
    elif fitType == "twoGaussians":
        testYs = TwoGaussiansForFit(testXs, *currentFitValues)
        gaussian1 = GaussianForFit(testXs, *(currentFitValues[0:3]))
        gaussian2 = GaussianForFit(testXs, *(currentFitValues[3:]))
        axTestHist.plot(testXs, gaussian1, color="r")
        axTestHist.plot(testXs, gaussian2, color="b")
    elif fitType == "PolyaPeak":
        testYs = PolyaForFit(testXs, polyaMean, currentFitValues[1], currentFitValues[2])
    elif fitType=="PolyaMean":
        testYs = PolyaForFit(testXs, *currentFitValues)
    elif fitType=="Adaptive":
        if len(currentFitValues)==3: # if single gaussian fitting worked
            testYs = GaussianForFit(testXs, *currentFitValues)
        else:
            testYs = TwoGaussiansForFit(testXs, *currentFitValues)
            gaussian1 = GaussianForFit(testXs, *(currentFitValues[0:3]))
            gaussian2 = GaussianForFit(testXs, *(currentFitValues[3:]))
            axTestHist.plot(testXs, gaussian1, color="r")
            axTestHist.plot(testXs, gaussian2, color="b")
    axTestHist.plot(testXs, testYs)

    axTestHist.axvline(currentFitValues[0], color="orange") 
    return 0




def FitIntegrations(integrationsToFit, numBins, printTestHistogram = False, returnCovariance = False, fitType = "gaussian"):
    if len(integrationsToFit) >= 20: # want a baseline number of pmt Integrations to actually fit to
        # Switch to absolute value for fitting (especially polya)
        integrationsToFit = numpy.abs(integrationsToFit)
        print(numpy.amin(integrationsToFit))
        
        minimumIntegrations = numpy.amin(integrationsToFit)
        maximumIntegrations = numpy.amax(integrationsToFit)
        IntegrationsDifference = maximumIntegrations - minimumIntegrations
        
        binLowerBound = minimumIntegrations-IntegrationsDifference*0.1
        if binLowerBound<0:
            binLowerBound = 0
        binUpperBound = maximumIntegrations+IntegrationsDifference*0.1
        binEdges = numpy.arange(binLowerBound, binUpperBound, step=IntegrationsDifference/numBins)
        
        # Possibly need to create a separate figure to store the hists in else it affects previous plots
        #histFigure = pyplot.figure(3)
        #histAx = histFigure.subplots(1,1)
        # currentHist = histAx.hist(combinedPMTPulseData[i], bins=binEdges)
        currentHist = numpy.histogram(integrationsToFit, bins=binEdges)
        #currentHist = pyplot.hist(combinedPMTPulseData[i], bins=binEdges)
        currentHistYs = currentHist[0]
        currentHistXs = currentHist[1]
        centeredXs = []
        for j in range(len(currentHistXs)-1):
            centeredXs.append( (currentHistXs[j]+currentHistXs[j+1]) /2 )
        
        centeredXsScaled = []
        for i in range(len(centeredXs)):
            centeredXsScaled.append(centeredXs[i]*10**12) # from V s to mV nS

        
        if fitType == "gaussian":
            currentFitValues, currentFitCovariance = FitGaussianIntegration(integrationsToFit, binEdges, currentHistXs, currentHistYs, centeredXs, centeredXsScaled, printTestHistogram)
       
        elif fitType == "twoGaussians":
            currentFitValues, currentFitCovariance = FitTwoGaussianIntegration(integrationsToFit, binEdges, currentHistXs, currentHistYs, centeredXs, centeredXsScaled, printTestHistogram)
            
        elif fitType == "PolyaPeak" or fitType=="PolyaMean":
            currentFitValues, currentFitCovariance, polyaMean = FitPolyaIntegration(integrationsToFit, binEdges, currentHistXs, currentHistYs, centeredXs, centeredXsScaled, printTestHistogram, fitType)
        
        
        elif fitType == "Adaptive":
            currentFitValues, currentFitCovariance = FitAdaptiveIntegration(integrationsToFit, binEdges, currentHistXs, currentHistYs, centeredXs, centeredXsScaled, printTestHistogram)
        
        else:
            raise Exception("Incorrect fitType.")
        
        #elif fitType == ""

        
        
        if printTestHistogram == True:
            printTestHistogramPlot(integrationsToFit, binEdges, currentHistXs, currentFitValues, fitType, polyaMean=0)
            



    if returnCovariance == True:
        perr = numpy.sqrt(numpy.diag(currentFitCovariance))
        # The covariance varies for the different parameters in the estimation
        # For a single gaussian I will have 3 parameters
        # Generally I only really care about the first gaussian anyway
        # There are 3 parameters then that I have to use to judge whether or not to try a two-gaussian fit
        # mean, sigma and amplitude
        # My initial thought is to keep things easy and just combine them all in quadrature
        # And then that way if even one of them is way off then it will sound the alarm
        print(perr)
        singleGaussianFitCovarianceCombined = numpy.sqrt( (perr[0])**2 + (perr[1])**2 + (perr[2])**2 )
        return currentFitValues, singleGaussianFitCovarianceCombined
            
            
    return currentFitValues




def MultiDiffuserAnalysis():
    numberOfPMTs = 96
    numBins = 40
    
    
    
    ### Add option to save/load combinedPMTPulseData
    
    analysisType = "integration"
    source = "lightBall"
    fileIdentifier = "800Pulse_HigherSuspendedLightBall"
    fitType="Adaptive"
    
    saveCombinedData = True
    useSavedData = True
    if useSavedData == True:
        combinedPMTPulseData_df = pd.read_csv(source + "_" + fileIdentifier + "_" + analysisType + "_savedPMTPulseData.csv", delimiter = ",")
        stacked = combinedPMTPulseData_df.stack().groupby(level=0).agg(list)
        combinedPMTPulseData_df["combined"] = stacked
        series = combinedPMTPulseData_df.loc[:, "combined"]
        combinedPMTPulseData = series.values
        #print(combinedPMTPulseData)
    else:
        if source == "diffuser":
            combinedPMTPulseData = GetMultiDiffuserPMTPulseData(numberOfPMTs, numBins, analysisType)
        elif source == "lightBall":
            combinedPMTPulseData = GetLightBallPMTPulseData(numberOfPMTs, numBins, analysisType)
    if saveCombinedData == True and useSavedData == False:
        outputList = []
        for i in range(len(combinedPMTPulseData)):
            outputList.append(combinedPMTPulseData[i])
        df = pd.DataFrame(outputList)
        df.to_csv(source + "_" + fileIdentifier + "_" + analysisType + "_savedPMTPulseData.csv", index=None)
        
        
    #print(combinedPMTPulseData.shape)




    
    
    # Don't want to store the parameters until I have the pmtMinima all combined, so all the files should be finished with by this point
    # Store the parameters for each pmtID, fitted to the combined histograms of all the diffusers put together
    fittedParameters = numpy.empty(numberOfPMTs, dtype="object")
    covariances = numpy.zeros(numberOfPMTs)
    
    for i in range(len(combinedPMTPulseData)):
        print(i)
        if analysisType == "minima":
            fittedParameters[i] = FitMinima(combinedPMTPulseData[i], numBins)
        elif analysisType == "integration":
            if i == 2:
                fittedParameters[i], covariances[i] = FitIntegrations(combinedPMTPulseData[i], numBins, printTestHistogram=True, returnCovariance=True, fitType=fitType)
            else:
                fittedParameters[i], covariances[i] = FitIntegrations(combinedPMTPulseData[i], numBins, returnCovariance=True, fitType=fitType)
        else:
            raise Exception("No analysisType for fitting.")

    pmtIDList = range(numberOfPMTs)
        
    
        
 
    pmtFittedMeans = []
    pmtFittedSigmas = []
    for i in range(len(fittedParameters)):
        #if fittedParameters[i].empty():
        #    pmtFittedMeans = 0
        #else:
        if type(fittedParameters[i]) != type(None):
            pmtFittedMeans.append(fittedParameters[i][0])
            pmtFittedSigmas.append(abs( fittedParameters[i][1] )) # There was one with a negative sigma for some reason, and it seems to just be mirrored - Negative sigma on a gaussian is squared anyway 
        else:
            pmtFittedMeans.append(0)
            pmtFittedSigmas.append(0)
       
    figFitAccuracy, axFitAccuracy = pyplot.subplots(1, 1, figsize=[32,16])
    axFitAccuracy.scatter(pmtIDList, pmtFittedMeans)
    if fitType in ["gaussian", "twoGaussians", "Adaptive"]:
        axFitAccuracy.errorbar(pmtIDList, pmtFittedMeans, pmtFittedSigmas)
    axFitAccuracy.set_xlabel("PMT ID")
    if analysisType == "minima":
        axFitAccuracy.set_ylabel("1pe Response (V)")
    elif analysisType == "integration":
        axFitAccuracy.set_ylabel(r"1pe Response (Vs$\Omega$)")
    # if analysisType == "minima":
    #     trueMeanHeight = -12.65 /1000
    #     trueHeightSpread = 6.522136474 /1000 / numpy.sqrt(2) # divide by sqrt 2 to fix the mistake in the gaussian from fitting
    # elif analysisType == "integration":
    #     trueMeanHeight = -5.2718409289646965e-11
    #     trueHeightSpread = 2.230580471991924e-11 # divide by sqrt 2 to fix the mistake in the gaussian from fitting
    trueMeanHeight, trueHeightSpread = GetTruthValues(analysisType, fitType)   
    axFitAccuracy.scatter(numberOfPMTs, trueMeanHeight, color="orange")
    if fitType in ["gaussian", "twoGaussians", "Adaptive"]:
        axFitAccuracy.errorbar(numberOfPMTs, trueMeanHeight, trueHeightSpread, color="orange")
    else:
        axFitAccuracy.axhline(trueMeanHeight, color="orange")
        
    pmtRelativeMeans = (numpy.array(pmtFittedMeans) - trueMeanHeight)/trueMeanHeight
    figFitPMTAccuracy, axFitPMTAccuracy = pyplot.subplots(1, 1, figsize=[32,16])
    axFitPMTAccuracy.scatter(pmtIDList, pmtRelativeMeans, s=100)
    axFitPMTAccuracy.set_xlabel("PMT ID")
    axFitPMTAccuracy.set_ylabel("Fractional Difference from True 1pe Response")
    axFitPMTAccuracy.axhline(0)
    
    
    figFitCov, axFitCov = pyplot.subplots(1, 1, figsize=[32,16])
    axFitCov.scatter(pmtIDList, covariances, marker="s", s=200)
    axFitCov.set_xlabel("PMT ID")
    axFitCov.set_ylabel("Fit Covariance P_err")
    #axFitCov.set_yscale("log")
    axFitCov.set_ylim(1,200)
    
    
    
    
    
    
    
    
    # Now want to try and quantify the accuracy of the thing with respect to the number of photons in
    # First get the number of events used for each and then plot that against some measure of maybe mean first?
    numberOfEventsInEach = []
    meanDifferences = []

    for i in range(len(combinedPMTPulseData)):
        numberOfEventsInEach.append(len(combinedPMTPulseData[i]))
        #print(pmtFittedMeans[i])
        meanDifferences.append(pmtFittedMeans[i] - trueMeanHeight)
        #print(meanDifferences[i])
        
    numberOfEventsInEach = numpy.array(numberOfEventsInEach)
    meanDifferences = numpy.array(meanDifferences)
    percentageMeanDifferences = meanDifferences / trueMeanHeight
    figMeanAccVsPhotonCount = pyplot.figure("MeanAccVsPhotonCountMulti", figsize=[32,16])
    axMeanAccVsPhotonCount = figMeanAccVsPhotonCount.subplots(1, 1)
    axMeanAccVsPhotonCount.scatter(numberOfEventsInEach, percentageMeanDifferences)
    axMeanAccVsPhotonCount.set_xlabel("Number of Hits")
    axMeanAccVsPhotonCount.set_ylabel("Fractional Difference from True Mean")
    #axMeanAccVsPhotonCount.set_ylim(-0.0035, 0.0035)
    axMeanAccVsPhotonCount.set_ylim(-0.1, 0.1)
    axMeanAccVsPhotonCount.set_xlim(0,5000)
    axMeanAccVsPhotonCount.axhline(0)   
    print("Total number of hits:",numpy.sum(numberOfEventsInEach))
    
    SplitLargeCounts(combinedPMTPulseData, numberOfPMTs, numBins, analysisType, fitType)
    
    
    #print("\n\nCalculating gain:")
    #print("True value gain:",GainFormula(trueMeanHeight*10**12))
    #print("Measured gain values:", GainFormula(numpy.array(pmtFittedMeans)*10**12))
    
    
    print("\n\nStarting randomness study:")
    
    InvestigateAccuracyRandomness(combinedPMTPulseData, numBins, analysisType, fitType)
    
   
    return 0
   
    
   
    
   
### SplitLargeCounts takes the top N most hit PMTs and then takes the mean and standard deviation of those to figure out how the accuracy changes with an increasing number of hits
### I'm thinking this might have been useful to check the PMT calibration laser time needed before, but now that that seems perfectly achievable, perhaps other things should be focussed on.
### The thing you really want to know is: How accurate is the calibration?
### If a PMT has a given number of hits in it, then you would want to link that to an accuracy of some sort.
### And then the accuracy of the calibration changes for different PMTs based on how many hits they receive, which is affected by the light source type.
### So to measure this and determine the accuracy, I would need
### A very accurate (well-populated, with lots of backing data) graph of accuracy vs number of hits
### And a good knowledge of how that data changes
### This means I need something like - 
### 1) Get a graph of accuracy (difference from true gain) vs numHits
### - This should be supported by a good number of PMTs, ideally with plenty of spare hits to sample from to avoid correlations in the higher numbers of hits (random sampling that I need to investigate)
### 2) Repeat this many times to get a real distribution of accuracies for each number of hits
### - If the method is unbiased, the accuracy should trend towards having 0 difference from the mean,
### - and while the method continues to improve with more input hits, the spread should decrease.
### This is kind of like what I have already, but more sophisticated and with lots more backing data.
### That may be the limiting problem here - I'm already getting 4Gb files with my current method
### It might be worth implementing something like the integration directly in the waveform processing (maybe there's even something in there already?)
### then just saving the output integration values rather than the full waveforms to decrease the file size and offload processing to the warwick system instead of mine
### Could also split into multiple different runs and combine the root files after if they're not too big for parallel processing of some of these events.


def GetTruthValues(analysisType, fitType):
    if analysisType == "minima":
        trueMeanHeight = -12.65 /1000
        trueHeightSpread = 6.522136474 /1000 / numpy.sqrt(2) # divide by sqrt 2 to fix the mistake in the gaussian from fitting
    elif analysisType == "integration":
        if fitType=="gaussian" or fitType=="twoGaussians":
            trueMeanHeight = -5.2718409289646965e-11
            trueHeightSpread = 2.230580471991924e-11 # divide by sqrt 2 to fix the mistake in the gaussian from fitting        
        elif fitType=="PolyaMean":
            trueMeanHeight = 5.424426626554663e-11
            trueHeightSpread = 6.367210320407468
        elif fitType=="PolyaPeak":
            trueMeanHeight = 4.706040446241892e-11
            trueHeightSpread = 6.367210320407468
        elif fitType=="Adaptive":
            trueMeanHeight = -5.2718409289646965e-11
            trueHeightSpread = 2.230580471991924e-11
    else:
        raise Exception("Invalid analysis type.")
    return abs(trueMeanHeight), abs(trueHeightSpread)




def InvestigateAccuracyRandomness(combinedPMTPulseData, numBins, analysisType, fitType):
    # The end goal is to get a mean + spread of the accuracy of a generic PMT when supplied with a given number of hits
    # To do this, you need MANY hits.
    # In fact, it would probably be better with just one PMT with millions of hits on it...
    # Is there a world in which I can combine all the PMTs' data and treat it as a single PMT?
    # The reason I think this might be possible is because the hits should be independent anyway...
    # Once a hit is registered in the PMT, it's the same regardless of which PMT it's in.
    # --- This is true at the moment because the PMTs all have the same properties/dark rates/etc.
    
    # Either way, you would then sample from that dataset and get the spreads at each number of hits
    # I'm thinking graphs of like differenceFromMean vs numHits, with multiple lines for different "pool" sizes
    # And the same thing with spreads
    # That should show me how the random sampling from the pool size affects the accuracy "spread"
    # And maybe I can figure out some of the true underlying spread from that?
    # this sounds good.
    
    trueMeanHeight, trueHeightSpread = GetTruthValues(analysisType, fitType)
    
    # First step is to load in the pmtHits data (integration or minima?) and combine everything into one pool
    singleListPMTPulseData = []
    for pmtHitList in combinedPMTPulseData:
        singleListPMTPulseData += pmtHitList
    print(len(singleListPMTPulseData))
    singleListPMTPulseData = numpy.array(singleListPMTPulseData)
    # print(singleListPMTPulseData[100000])
    # This means we now have a list of integration values
    
    # Next is to sample from these, form a histogram, fit it, and compare with the true values to find the accuracy.
    # This process needs to be repeated multiple times for each sample size to determine the spread for the mean and the spread for the spread
    numberOfHitsToTest = numpy.arange(1000, 100000, step=1000)
    poolSizes = [5000, 10000, 25000, 50000, 100000, 200000]#, 400000, 600000]
    numRepeats = 10 # The number of times to pick random samples from the pool
    
    # meanData contains multiple lists, each of which is a list of mean differences for each number of hits, and each point on that graph is an average of [numRepeats] runs 
    meanData = []
    sigmaData = []
    meanStdDevData = []
    sigmaStdDevData = []
    
    
    for p in range(len(poolSizes)):
        print("Investigating pool size",p)
        trimmedDataPool = numpy.random.choice(singleListPMTPulseData,size=poolSizes[p], replace=False)
        
        # currentPoolMeanMeans contains the mean differences for each number of hits in the current pool, where each point is the average of [numRepeats] runs
        currentPoolMeanMeans = []
        # currentPoolStdDevMeans contains the standard deviation of the mean differences for each number of hits in the current pool, calculated from [numRepeats] runs
        currentPoolStdDevMeans = []
        # Same as currentPoolMeanMeans but for the sigma values of the distribution
        currentPoolMeanSigmas = []
        # Same as currentPoolStdDevMeans but for the sigma values of the distribution
        currentPoolStdDevSigmas = []
        for i in range(len(numberOfHitsToTest)):
            currentMeans = []
            currentSigmas = []
            for j in range(numRepeats):
                if numberOfHitsToTest[i] > poolSizes[p]:
                    continue
                randomSample = numpy.random.choice(trimmedDataPool, size=numberOfHitsToTest[i], replace=False)
                if analysisType == "minima":
                    fitValues = FitMinima(randomSample, numBins)
                elif analysisType == "integration":
                    fitValues = FitIntegrations(randomSample, numBins, fitType=fitType)
                else:
                    raise Exception("Incorrect analysis type.")
                if type(fitValues) != type(None):
                    currentMeans.append(fitValues[0])
                    currentSigmas.append(abs(fitValues[1]))
                else:
                    raise Exception("The fit has failed.")  
            currentMeans = numpy.array(currentMeans)
            currentSigmas = numpy.array(currentSigmas)
            currentMeanDiffs = currentMeans - trueMeanHeight
            currentSigmaDiffs = currentSigmas - trueHeightSpread
            currentFractionalMeanDifferences = currentMeanDiffs / trueMeanHeight 
            currentFractionalSigmaDifferences = currentSigmaDiffs / trueHeightSpread
            # At this point we have [numRepeats] copies of different means and sigmas for this number of hits
            # Want to probably use the mean meanDiff and the mean sigmaDiff, along with error bars for each of those
            
            currentPoolMeanMeans.append( numpy.average(currentFractionalMeanDifferences) )
            currentPoolStdDevMeans.append ( numpy.std(currentFractionalMeanDifferences) )
            currentPoolMeanSigmas.append ( numpy.average(currentFractionalSigmaDifferences) )
            currentPoolStdDevSigmas.append( numpy.std(currentFractionalSigmaDifferences) )
        
        meanData.append(currentPoolMeanMeans)
        meanStdDevData.append(currentPoolStdDevMeans)
        
        sigmaData.append(currentPoolMeanSigmas)
        sigmaStdDevData.append(currentPoolStdDevSigmas)
        
        
    # Means are the ones I care about the most
    # Next plot things:
    # Starting with the means, for each pool size, need to get the meanDifferences for each number of hits
    figCombinedPoolsMeans, axCombinedPoolsMeans = pyplot.subplots(1, 1, figsize=[32,16])
    figCombinedPoolsSigmas, axCombinedPoolsSigmas = pyplot.subplots(1, 1, figsize=[32,16])
    
    figCombinedPoolsMeansClean, axCombinedPoolsMeansClean = pyplot.subplots(1, 1, figsize=[32,16])
    figCombinedPoolsSigmasClean, axCombinedPoolsSigmasClean = pyplot.subplots(1, 1, figsize=[32,16])
    
    for p in range(len(poolSizes)):
        #axCombinedPoolsMeans.plot(numberOfHitsToTest[:len(meanData[p])], meanData[p], label = poolSizes[p])
        axCombinedPoolsMeans.errorbar(numberOfHitsToTest[:len(meanData[p])], meanData[p], meanStdDevData[p], label=poolSizes[p])
        axCombinedPoolsSigmas.errorbar(numberOfHitsToTest[:len(sigmaData[p])], sigmaData[p], sigmaStdDevData[p], label = poolSizes[p])
        
        
    axCombinedPoolsMeans.legend()
    axCombinedPoolsSigmas.legend()
    
    axCombinedPoolsMeans.set_xlabel("Number of Hits")
    axCombinedPoolsMeans.set_ylabel("(Fit Mean - True Mean) / True Mean")
    axCombinedPoolsSigmas.set_xlabel("Number of Hits")
    axCombinedPoolsSigmas.set_ylabel("(Fit Sigma - True Sigma) / True Sigma")
    
    
    
    axCombinedPoolsMeansClean.errorbar(numberOfHitsToTest[:len(meanData[-1])], meanData[-1], meanStdDevData[-1], label=poolSizes[-1])
    axCombinedPoolsSigmasClean.errorbar(numberOfHitsToTest[:len(sigmaData[-1])], sigmaData[-1], sigmaStdDevData[-1], label = poolSizes[-1])
        
    axCombinedPoolsMeansClean.set_xlabel("Number of Hits")
    axCombinedPoolsMeansClean.set_ylabel("(Fit Mean - True Mean) / True Mean")
    axCombinedPoolsSigmasClean.set_xlabel("Number of Hits")
    axCombinedPoolsSigmasClean.set_ylabel("(Fit Sigma - True Sigma) / True Sigma")
    
    
    return 0







def SplitLargeCounts(combinedPMTPulseData, numberOfPMTs, numBins, analysisType, fitType):
    """Split the counts for the most-hit pmts into smaller chunks to get smooth graphs with controlled numbers of PMT hits.
    This can then be used to make a graph of fractional error vs number of hits."""
    
    # Need to load in the pmtMinima, then take the largest data sets as a basis.
    # Then split up these large datasets into smaller bits, fit each independently and plot the results.
    # Repeat for the n most hit pmts and get some kind of estimate of an error bar on that.
    
    numUsedMostHitPMTs = 10 # use the 8 most hit pmts for the analysis.
    
    indicesOfMostHitPMTs = []
    
    # Use numpy argsort to give the indices of the longest datasets.
    combinedPMTPulseDataLengths = []
    for i in range(len(combinedPMTPulseData)):
        combinedPMTPulseDataLengths.append(len(combinedPMTPulseData[i]))
    
    # Now we have an array containing the lengths of each dataset at the corresponding pmtID index.
    # use numpy argsort on this to give the indices that have the longest datasets.
    
    sortedCombinedPMTPulseDataLengthsIndices = numpy.argsort(combinedPMTPulseDataLengths)
    indicesOfMostHitPMTs = sortedCombinedPMTPulseDataLengthsIndices[-numUsedMostHitPMTs:]
    print(indicesOfMostHitPMTs)
    for index in indicesOfMostHitPMTs:
        print(index, combinedPMTPulseDataLengths[index])
    
    numberOfHitsToTest = [100, 250, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]
    
    
    fractionalDifferencesFromTrueMean = []
    fractionalDifferencesFromTrueSigma = []
    # if analysisType == "minima":
    #     trueMeanHeight = -12.65 /1000
    #     trueHeightSpread = 6.522136474 /1000 / numpy.sqrt(2) # divide by sqrt 2 to fix the mistake in the gaussian from fitting
    # elif analysisType == "integration":
    #     trueMeanHeight = -5.2718409289646965e-11
    #     trueHeightSpread = 2.230580471991924e-11 # divide by sqrt 2 to fix the mistake in the gaussian from fitting
    trueMeanHeight, trueHeightSpread = GetTruthValues(analysisType, fitType)  
    
    # for each number of hits to test, want to trim the dataset to this (with random sampling or just straight up taking the first n values?)
    for numberOfHits in numberOfHitsToTest:
        currentFractionalMeanDifferences = []
        currentFractionalSigmaDifferences = []
        
        for i in range(numUsedMostHitPMTs):
            # Loop over the PMTs and trim them, make a histogram for the new dataset, then fit.
            trimmedDataset = numpy.random.choice(combinedPMTPulseData[indicesOfMostHitPMTs[i]], size=numberOfHits, replace=False)
            
            if analysisType == "minima":
                fitValues = FitMinima(trimmedDataset, numBins)
            elif analysisType == "integration":
                fitValues = FitIntegrations(trimmedDataset, numBins, fitType=fitType)
            else:
                raise Exception("Incorrect analysis type.")
            
            if type(fitValues) != type(None):
                fittedMean = fitValues[0]
                fittedSigma = abs(fitValues[1])
            else:
                raise Exception("The fit has failed.")
            
            differenceInMean = fittedMean - trueMeanHeight
            differenceInSigma = fittedSigma - trueHeightSpread
            
            currentFractionalMeanDifferences.append(differenceInMean / trueMeanHeight) 
            currentFractionalSigmaDifferences.append(differenceInSigma / trueHeightSpread)
        fractionalDifferencesFromTrueMean.append(currentFractionalMeanDifferences)
        fractionalDifferencesFromTrueSigma.append(currentFractionalSigmaDifferences)


    averagedMeanFractionalDifferences = []
    meanFractionalDifferenceErrors = []
    averagedSigmaFractionalDifferences = []
    sigmaFractionalDifferenceErrors = []
    
    for i in range(len(fractionalDifferencesFromTrueMean)):
        averagedMeanFractionalDifferences.append( numpy.average(fractionalDifferencesFromTrueMean[i]) )
        meanFractionalDifferenceErrors.append( numpy.std(fractionalDifferencesFromTrueMean[i]) )
        averagedSigmaFractionalDifferences.append( numpy.average(fractionalDifferencesFromTrueSigma[i]) )
        sigmaFractionalDifferenceErrors.append( numpy.std(fractionalDifferencesFromTrueSigma[i]) )
    
    figAccuracyVsNumHitsMean, axAccuracyVsNumHitsMean = pyplot.subplots(1, 1, figsize=[32,16])
    axAccuracyVsNumHitsMean.scatter(numberOfHitsToTest, numpy.array(averagedMeanFractionalDifferences)*100)
    axAccuracyVsNumHitsMean.errorbar(numberOfHitsToTest, numpy.array(averagedMeanFractionalDifferences)*100, numpy.array(meanFractionalDifferenceErrors)*100)
    axAccuracyVsNumHitsMean.set_xlabel("Number of Hits")
    axAccuracyVsNumHitsMean.set_ylabel("Percentage Difference from True Mean")
    axAccuracyVsNumHitsMean.axhline(0)
    
    figAccuracyVsNumHitsSigma, axAccuracyVsNumHitsSigma = pyplot.subplots(1, 1, figsize=[32,16])
    axAccuracyVsNumHitsSigma.scatter(numberOfHitsToTest, numpy.array(averagedSigmaFractionalDifferences)*100)
    axAccuracyVsNumHitsSigma.errorbar(numberOfHitsToTest, numpy.array(averagedSigmaFractionalDifferences)*100, numpy.array(sigmaFractionalDifferenceErrors)*100)
    axAccuracyVsNumHitsSigma.set_xlabel("Number of Hits")
    axAccuracyVsNumHitsSigma.set_ylabel("Percentage Difference from True Sigma")
    axAccuracyVsNumHitsSigma.axhline(0)
    
    return 0


if __name__ == "__main__":
    MultiDiffuserAnalysis()
    #TestDoubleGaussian()
    #TestCombinedModelForFit()
    #TestPolyaForFit()
    
    # xs = numpy.arange(20, 260, step=0.1)
    # ys = 2000*numpy.exp(-0.01*xs)
    # fig,ax = pyplot.subplots(1,1, figsize=[32,16])
    # ax.set_yscale("log")
    # ax.plot(xs, ys)
    # ax.set_ylim(100,2000)
    
    
    
    
    
    
    



    
    