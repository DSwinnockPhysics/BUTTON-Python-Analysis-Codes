import uproot
import numpy
import matplotlib.pyplot as pyplot
import awkward as ak
import pandas as pd
import scipy
pyplot.rcdefaults()
pyplot.rcParams["font.size"]=36
#pyplot.rcParams["figure.dpi"]=70




def GaussianForFit(x, mean, sigma, amplitude):
    return amplitude * numpy.exp(-1 * ((x-mean)/(numpy.sqrt(2)*sigma))**2)



def TwoGaussiansForFit(x, mean1, sigma1, amplitude1, mean2, sigma2, amplitude2):
    """Uses two gaussians with separate parameters and combines them."""
    gaussian1 = amplitude1 * numpy.exp(-1 * ((x-mean1)/(numpy.sqrt(2)*sigma1))**2)
    gaussian2 = amplitude2 * numpy.exp(-1 * ((x-mean2)/(numpy.sqrt(2)*sigma2))**2)
    return gaussian1 + gaussian2



def samplesToTime(sample):
    sampleRate = 500e6
    return sample/sampleRate

def timeToSamples(time):
    sampleRate = 500e6
    return time*sampleRate


# def GainFormula(singlePECharge):
#     # Not to be trusted, needs a bit more careful thinking about.
#     electronCharge = 1.602176634e-19 # Coulombs
#     impedance = 50 # Ohms
#     ampGain = 1# 10 # 1 for ratpac sim, 10 in the paper they used # This is the amplifier gain post-pmt
#     gain = (2 * singlePECharge * 10**-12) / (electronCharge * ampGain * impedance)
#     return gain


    
def GetDigitiserProperties():
    vHigh = 1000 # mV
    vLow = -1000 # mV
    vOffset = 800 # mV
    adcCountsPerMilliV = 16384 / (vHigh-vLow) 
    sampleRate = 500e6 # 500MS/s
    digitiserProperties = {"vHigh":vHigh,
                          "vLow":vLow,
                           "vOffset":vOffset,
                           "adcCountsPerMilliV":adcCountsPerMilliV,
                           "sampleRate":sampleRate
                           }
    return digitiserProperties
    




def FitGaussianIntegration(integrationsToFit, binEdges, currentHistXs, currentHistYs, centeredXs, centeredXsScaled, printTestHistogram):
    """Takes the integrations to fit (for setting the x axis scale), and the histogram of the integration results.
    Fits a Gaussian to the histogram."""
    integrationFitParametersGuess = [-5e-11, 2e-11, 12]
    # Want to change these initial guesses into something based on the waveform for better starting parameters
    # Would be good to know the maximum value of the histogram and the std dev
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
    print("Initial guess parameters:",integrationFitParametersGuess)
    
    if printTestHistogram == True:
        testXs = numpy.arange(currentHistXs[0], currentHistXs[-1], step=(currentHistXs[1]-currentHistXs[0])/100)
        figInitial, axInitial = pyplot.subplots(1, 1, figsize=[32,16])
        axInitial.hist(integrationsToFit, bins=binEdges)
        axInitial.plot(testXs, GaussianForFit(testXs, *integrationFitParametersGuess), color="r")
        axInitial.axvline(GetTruthValues()[0], color="orange")
        
    # Possibly scale the parameters depending on what the units of the data are
    # Ratpac tends to use mV and ns instead of V and s resulting in factors of 10^12
    # Ideally all the fit parameters are approximately the same order of magnitude for better fitting
    # and better error (covariance) estimation.
    scaledIntegrationFitParametersGuess = [integrationFitParametersGuess[0],#*10**12,
                                           integrationFitParametersGuess[1],#*10**12,
                                           integrationFitParametersGuess[2]
                                           ]
    
    
    #print(centeredXsScaled)
    #print(currentHistYs)
    
    
    currentFitValues, currentFitCovariance = scipy.optimize.curve_fit(GaussianForFit,
                                                                      centeredXsScaled, 
                                                                      currentHistYs, 
                                                                      #p0=integrationFitParametersGuess
                                                                      p0=scaledIntegrationFitParametersGuess
                                                                      )
    
    print("Fit values before scaling:",currentFitValues)
    print("Covariance:",currentFitCovariance)
    # Un-scale the fit values (if they were scaled)
    currentFitValues = [currentFitValues[0],#*10**-12,
                        currentFitValues[1],#*10**-12,
                        currentFitValues[2]
                        ]
    
    if printTestHistogram == True:
        figFitted, axFitted = pyplot.subplots(1, 1, figsize=[32,16])
        axFitted.hist(integrationsToFit, bins=binEdges)
        axFitted.plot(testXs, GaussianForFit(testXs, *currentFitValues), color="r")
        
        
    return currentFitValues, currentFitCovariance
            


def FitTwoGaussianIntegration(integrationsToFit, binEdges, currentHistXs, currentHistYs, centeredXs, centeredXsScaled, printTestHistogram):
    """Takes the integrations to fit (for setting the x axis scale), and the histogram of the integration results.
    Fits a combination of two Gaussians to the histogram."""
    
    firstGaussianGuessParameters = [-5e-11, 2e-11, 100]
    secondGaussianGuessParameters = [-1.2e-10, 3e-11, 10]
    
    # Want to change these initial guesses into something based on the waveform for better starting parameters
    # Would be good to know the maximum value of the histogram and the std dev
    
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
    #firstGaussianGuessParameters = [histogramMean*0.8, 0.8*histogramStdDev, histogramMax]
    #secondGaussianGuessParameters = [1.6*histogramMean, 0.8*histogramStdDev, histogramMax/3]
    firstGaussianGuessParameters = [histogramMean, histogramStdDev, histogramMax]
    
    # Make second guess parameters based on first - the 2pe influence corresponds (sort of) to a gaussian
    # that has a mean around twice that of the 1pe response, with a lower peak 
    # This is reflected here with somewhat arbitrary scaling but the fitting typically fixes this anyway.
    secondGaussianGuessParameters = [2*histogramMean, histogramStdDev, histogramMax/3]
    
    combinedGuessParameters = firstGaussianGuessParameters + secondGaussianGuessParameters
    if printTestHistogram == True:
        testXs = numpy.arange(currentHistXs[0], currentHistXs[-1], step=(currentHistXs[1]-currentHistXs[0])/100)
        figInitial, axInitial = pyplot.subplots(1, 1, figsize=[32,16])
        axInitial.hist(integrationsToFit, bins=binEdges)
        axInitial.plot(testXs, TwoGaussiansForFit(testXs, *firstGaussianGuessParameters, *secondGaussianGuessParameters))
        axInitial.plot(testXs, GaussianForFit(testXs, *firstGaussianGuessParameters), color="r")
        axInitial.plot(testXs, GaussianForFit(testXs, *secondGaussianGuessParameters), color="b")
    
    # The parameters should be scaled such that they are all roughly the same order of magnitude, otherwise one variable can dominate the regression 
    

    scaledGuessParameters = [combinedGuessParameters[0],#*10**12,
                             combinedGuessParameters[1],#*10**12,
                             combinedGuessParameters[2],
                             combinedGuessParameters[3],#*10**12,
                             combinedGuessParameters[4],#*10**12,
                             combinedGuessParameters[5]
                             ]
    
    # m1, s1, a1, m2, s2, a2
    # This locks the standard deviation and amplitude of the gaussians to be positive only
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
    currentFitValues = [currentFitValues[0],#*10**-12,
                        currentFitValues[1],#*10**-12,
                        currentFitValues[2],
                        currentFitValues[3],#*10**-12,
                        currentFitValues[4],#*10**-12,
                        currentFitValues[5]
                        ]
    
    
    if printTestHistogram == True:
        figFitted, axFitted = pyplot.subplots(1, 1, figsize=[32,16])
        axFitted.hist(integrationsToFit, bins=binEdges)
        axFitted.plot(testXs, TwoGaussiansForFit(testXs, *currentFitValues[0:3], *currentFitValues[3:]))
        axFitted.plot(testXs, GaussianForFit(testXs, *currentFitValues[0:3]), color="r")
        axFitted.plot(testXs, GaussianForFit(testXs, *currentFitValues[3:]), color="b")
    return currentFitValues, currentFitCovariance
        






def FitIntegrations(integrationsToFit, numBins, printTestHistogram = False, returnCovariance = False, fitType = "gaussian"):
    """Takes the integrations array and manages which type of fitting to perform."""
    if len(integrationsToFit) >= 20: # want a baseline number of pmt Integrations to actually fit to
        # Switch to absolute value for fitting
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
        

        # Create the histogram 
        currentHist = numpy.histogram(integrationsToFit, bins=binEdges)

        
        currentHistYs = currentHist[0]
        currentHistYs = currentHistYs.astype(numpy.float64)
        # maxHistY = numpy.amax(currentHistYs)
        # print("Max hist Y:", maxHistY)
        # for y in range(len(currentHistYs)):
        #     print(currentHistYs[y])
        #     currentHistYs[y] = currentHistYs[y] / maxHistY
        #     print(currentHistYs[y])
        print("Current hist ys:", currentHistYs)
        currentHistXs = currentHist[1]
        centeredXs = []
        for j in range(len(currentHistXs)-1):
            centeredXs.append( (currentHistXs[j]+currentHistXs[j+1]) /2 )
        
        centeredXsScaled = []
        # maxXValue = max(centeredXs)
        # Depending on the units of the data, scale the Xs so that the mean/std dev are close to the amplitude in magnitude
        for i in range(len(centeredXs)):
            centeredXsScaled.append(centeredXs[i])# *10**12) # from V s to mV nS
        print("Centered Xs Scaled:", centeredXsScaled)
        
        if fitType == "gaussian":
            currentFitValues, currentFitCovariance = FitGaussianIntegration(integrationsToFit, binEdges, currentHistXs, currentHistYs, centeredXs, centeredXsScaled, printTestHistogram)
       
        elif fitType == "twoGaussians":
            currentFitValues, currentFitCovariance = FitTwoGaussianIntegration(integrationsToFit, binEdges, currentHistXs, currentHistYs, centeredXs, centeredXsScaled, printTestHistogram)
            
        # elif fitType == "PolyaPeak" or fitType=="PolyaMean":
        #     currentFitValues, currentFitCovariance, polyaMean = FitPolyaIntegration(integrationsToFit, binEdges, currentHistXs, currentHistYs, centeredXs, centeredXsScaled, printTestHistogram, fitType)
        
        
        elif fitType == "Adaptive":
            currentFitValues, currentFitCovariance = FitAdaptiveIntegration(integrationsToFit, binEdges, currentHistXs, currentHistYs, centeredXs, centeredXsScaled, printTestHistogram)
        
        else:
            raise Exception("Incorrect fitType.")
        
        #elif fitType == ""

        
        
        # if printTestHistogram == True:
        #     printTestHistogramPlot(integrationsToFit, binEdges, currentHistXs, currentFitValues, fitType, polyaMean=0)
            



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











def FitAdaptiveIntegration(integrationsToFit, binEdges, currentHistXs, currentHistYs, centeredXs, centeredXsScaled, printTestHistogram):
    """Fits the data using a single Gaussian, then if the covariance is too high it performs a two-gaussian fit."""
    # Uses one method, then fits with another if that one fails to fit based on the covariance
    # Start with single gaussian
    firstFitValues, firstFitCovariance = FitGaussianIntegration(integrationsToFit, binEdges, currentHistXs, currentHistYs, centeredXs, centeredXsScaled, printTestHistogram)
    
    # Covariance needs some tuning - or some way of making a good threshold without knowing exactly what data is coming in or biasing things
    # It can depend on the number of hits because of the amplitude scale
    # Some adaptive scaling on the amplitude may help - something like normalising based on the number of events?
    # That way it wouldn't be biased and would be the same for every histogram
    # And is easily implementable for real data as well (?)
    # Then, the amplitude scale, mean+stdDev scale would all be relatively fixed in terms of order of magnitude
    # With only the real individual difference between them appearing
    # Which might make the covariance a parameter that can remain more static
    
    # This is awkward to set automatically when there are lots of different files being loaded and compared,
    # although maybe it's fine to do it per file?  That way it would automatically adapt if one has a run of 100k and another has 1M for example, maybe that's good?
    # I'd need to get a number of pulses from the file and feed that to here
    # totalNumEvents = 100000 # Needs to be a parameter that is closer to the actual loading of data or better, tied to it, so that it can be done either more easily or automatically.
    fitCovarianceThreshold = 250 # / totalNumEvents # This needs to have some kind of good justification on how to be set when you don't know the true value of things.
    
    
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



def GetTruthValues(): # (analysisType, fitType):
    # if analysisType == "minima":
    #     trueMeanHeight = -12.65 /1000
    #     trueHeightSpread = 6.522136474 /1000 / numpy.sqrt(2) # divide by sqrt 2 to fix the mistake in the gaussian from fitting
    # elif analysisType == "integration":
    #     if fitType=="gaussian" or fitType=="twoGaussians":
    #         trueMeanHeight = -5.2718409289646965e-11
    #         trueHeightSpread = 2.230580471991924e-11 # divide by sqrt 2 to fix the mistake in the gaussian from fitting        
    #     elif fitType=="PolyaMean":
    #         trueMeanHeight = 5.424426626554663e-11
    #         trueHeightSpread = 6.367210320407468
    #     elif fitType=="PolyaPeak":
    #         trueMeanHeight = 4.706040446241892e-11
    #         trueHeightSpread = 6.367210320407468
    #     elif fitType=="Adaptive":
    #         trueMeanHeight = -5.2718409289646965e-11
    #         trueHeightSpread = 2.230580471991924e-11
    # else:
    #     raise Exception("Invalid analysis type.")
    
    
    # Currently only using adaptive fitting, so return those truth values
    # These are obtained from a little simulation of the ratpac fundamental charge values
    # I'm somewhat sure that the true mean should be at 50 mV ns Ohm but I didn't quite realise this at the time I was checking the ratpac code
    # So that may not be true, it'd need looking at
    # It's very close anyway with the simulation results.
    # This may also change if the experimental setup is different than ratpac's processing, so watch out for that.
    trueMeanHeight = 4.9987991717775634e-11 * 10**12 # -5.2718409289646965e-11 * 10**12
    trueHeightSpread = 1.89939761243927e-11 * 10**12 # 2.230580471991924e-11 * 10**12
    
    return abs(trueMeanHeight), abs(trueHeightSpread)




def GetCombinedPMTPulseDataFromFile(numberOfPMTs, sourceType, fileName):
    """Takes the data from the ratpac outntuple and rearranges it.
    The output is an array of length numberOfPMTs, for which each element is a list of the integrated charge values, one for each hit on that PMT."""
    #combinedPMTPulseData = numpy.empty(numberOfPMTs, dtype="object")
    # Create result array
    combinedPMTPulseData = [[] for _ in range(96)]
    for i in range(len(combinedPMTPulseData)):
        combinedPMTPulseData[i] = []
        
    with uproot.open(fileName) as file: # "lightBallCalibration3_150Photons_120kEvents.ntuple.root"
        print(file.keys())
        print(file["output"].keys())
        output = file["output"]

        integrationsList = output["fit_FOM_ButtonWave_integration"].array()
        print(range(len(integrationsList)))
        
        # The old way was just a nested for loop but it was too slow to run on the 10M event data
        # So there's a new way beneath that is more efficient.
        #for integrationList in output["fit_FOM_ButtonWave_integration"].array():
        # for triggeredEventNumber in range(len(output["fit_FOM_ButtonWave_integration"].array())):
        #     print(triggeredEventNumber)
        #     integrationList = output["fit_FOM_ButtonWave_integration"].array()[triggeredEventNumber]
        #     pmtIDList = output["digitPMTID"].array()[triggeredEventNumber]
            
        #     if len(integrationList) == 0:
        #         #print("Empty")
        #         continue
        #     else:
        #         #print(integrationList)
        #         for i in range(len(integrationList)):
        #             currentPMTID = pmtIDList[i]
        #             combinedPMTPulseData[currentPMTID].append(integrationList[i])
        
        print()
        pmtIDList = output["digitPMTID"].array()
        
        print("Concatenating indexes:")
        # Flatten the nested lists
        flat_indexes = numpy.concatenate([numpy.array(x) for x in pmtIDList])
        print("Concatenated indexes")
        flat_values = numpy.concatenate([numpy.array(x) for x in integrationsList])
        print("Concatenated values")

        i=0
        # Populate using the flattened arrays
        for idx, val in zip(flat_indexes, flat_values):
            #print(i)
            combinedPMTPulseData[idx].append(val)
            i+=1
        
    return combinedPMTPulseData



def CalibrationAnalysis(fileName, makeSingleRunFigures = True):
    numberOfPMTs = 96
    numBins = 30
    
    ### Add option to save/load combinedPMTPulseData
    # analysisType = "integration" # Analysis type is always integration now
    source = "lightBall" # will be useful for diffuser later on
    # fileName =
    fileIdentifier = "Data/" + fileName + ".ntuple.root" # just a file name
    # fitType="Adaptive" # Fit type is always adaptive now





    useSavedData = True
    saveCombinedData = False
    
    if useSavedData == True:
        combinedPMTPulseData_df = pd.read_csv("Data/" + fileName + "_" + source + "_savedPMTPulseData.csv", delimiter = ",")
        stacked = combinedPMTPulseData_df.stack().groupby(level=0).agg(list)
        combinedPMTPulseData_df["combined"] = stacked
        series = combinedPMTPulseData_df.loc[:, "combined"]
        combinedPMTPulseData = series.values
        #print(combinedPMTPulseData)
    else:
        combinedPMTPulseData = GetCombinedPMTPulseDataFromFile(numberOfPMTs, source, fileIdentifier) # Want to get a list of arrays, where the ID of the list is the PMT number and the list is a list of integration values(?)
    # Maybe save organised combinedPMTPulseData to avoid having to process it each time?
    # May be more important for the 10x case but still

    if saveCombinedData == True and useSavedData == False:
        outputList = []
        for i in range(len(combinedPMTPulseData)):
            outputList.append(combinedPMTPulseData[i])
        df = pd.DataFrame(outputList)
        df.to_csv("Data/" + fileName + "_" + source + "_savedPMTPulseData.csv", index=None)
        
    
    # print(len(combinedPMTPulseData))
    # for i in range(len(combinedPMTPulseData)):
    #     print(combinedPMTPulseData[i])
    print("Got combined PMT pulse data")
    # I also want a number of pulses, how 9especially if the data is loaded from the saved integrations, which give no indication of this?

    fittedParameters = numpy.empty(numberOfPMTs, dtype="object")
    covariances = numpy.zeros(numberOfPMTs)
    
    for i in range(len(combinedPMTPulseData)):
        print(i)
        if i == 12:
            fittedParameters[i], covariances[i] = FitIntegrations(combinedPMTPulseData[i], numBins, printTestHistogram=True, returnCovariance=True, fitType="Adaptive")
        else:
            fittedParameters[i], covariances[i] = FitIntegrations(combinedPMTPulseData[i], numBins, returnCovariance=True, fitType="Adaptive")

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
       
    
    
    
    
    if makeSingleRunFigures == True:
        figFitAccuracy, axFitAccuracy = pyplot.subplots(1, 1, figsize=[32,16])
        axFitAccuracy.scatter(pmtIDList, pmtFittedMeans)
        #if fitType in ["gaussian", "twoGaussians", "Adaptive"]:
        axFitAccuracy.errorbar(pmtIDList, pmtFittedMeans, pmtFittedSigmas, fmt='')
        axFitAccuracy.set_xlabel("PMT ID")
        trueMeanHeight, trueHeightSpread = GetTruthValues()   
        axFitAccuracy.scatter(numberOfPMTs, trueMeanHeight, color="orange")
        #if fitType in ["gaussian", "twoGaussians", "Adaptive"]:
        axFitAccuracy.errorbar(numberOfPMTs, trueMeanHeight, trueHeightSpread, color="orange")
        #else:
        #    axFitAccuracy.axhline(trueMeanHeight, color="orange")
        axFitAccuracy.set_ylabel(r"1pe Integrated Response (mV ns $\Omega$)")
        
        
        
        
        
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
    
    
    
    return pmtFittedMeans, pmtFittedSigmas
    
    
    

def GetPMTHitCounts(fileName):
    fileIdentifier = "Data/" + fileName + ".ntuple.root" # just a file name
    with uproot.open(fileIdentifier) as file: # "lightBallCalibration3_150Photons_120kEvents.ntuple.root"
        output = file["output"]
        pmtIDList = output["digitPMTID"].array()
        flat_indexes = numpy.concatenate([numpy.array(x) for x in pmtIDList])
        print("Add photon hits:",len(flat_indexes))
        uniquePMTs, pmtCounts = numpy.unique(flat_indexes, return_counts=True)
    
        
    return pmtCounts
    
    
    

    
    
    
if __name__ == "__main__":
    
    channel_numbers = [148,97,157,153,155,130,171,29,6,146,61,107,9,47,134,92,167,26,14,102,135,59,160,103,154,150,169,75,16,27,94,998,132,83,76,7,78,143,166,88,84,138,49,96,48,37,67,163,53,12,10,87,34,42,33,104,15,105,142,30,164,82,81,90,32,57,74,133,147,161,63,66,149,3,145,106,999,108,28,98,141,159,50,136,1,54,51,43,71,111,55,65,56,112,162,73]
    print(len(channel_numbers))
    
    
    
    pmtFittedMeansList = []
    pmtFittedSigmasList = []
    fileNames = [
        "integrationTest_1MEvents_1",
        "integrationTest_1MEvents_2",
        "integrationTest_1MEvents_3",
        "integrationTest_1MEvents_4",
        "integrationTest_1MEvents_5",
        "integrationTest_1MEvents_6",
        "integrationTest_1MEvents_7",
        "integrationTest_1MEvents_8",
        "integrationTest_1MEvents_9",
        "integrationTest_1MEvents_10"
        ]
    
    for fileName in fileNames:
        print("        Processing file:",fileName)
        pmtFittedMeansToAdd, pmtFittedSigmasToAdd = CalibrationAnalysis(fileName, makeSingleRunFigures=False)
        pmtFittedMeansList.append(pmtFittedMeansToAdd)
        pmtFittedSigmasList.append(pmtFittedSigmasToAdd)
    
    print("Length of all pmtFittedMeansList:", len(pmtFittedMeansList), len(pmtFittedSigmasList))

    numberOfPMTs = 96
    pmtIDList = range(numberOfPMTs)
    
    pmtFittedMeansList = numpy.array(pmtFittedMeansList)
    pmtFittedSigmasList = numpy.array(pmtFittedSigmasList)
    
    
    
    
    averagePMTFittedMeans = numpy.average(pmtFittedMeansList,axis=0)
    stdDevPMTFittedMeans = numpy.std(pmtFittedMeansList, axis=0)
    #averageMPTFittedStdDevs = numpy.average(pmtFittedSigmasList,axis=1)
    
    print(len(averagePMTFittedMeans))
    print(len(stdDevPMTFittedMeans))
        
    

    figCombinedFitAccuracy, axCombinedFitAccuracy = pyplot.subplots(1, 1, figsize=[32,16])
    axCombinedFitAccuracy.scatter(channel_numbers, averagePMTFittedMeans)
    #if fitType in ["gaussian", "twoGaussians", "Adaptive"]:
    axCombinedFitAccuracy.errorbar(channel_numbers, averagePMTFittedMeans, stdDevPMTFittedMeans, fmt=' ')
    axCombinedFitAccuracy.set_xlabel("PMT ID")
    trueMeanHeight, trueHeightSpread = GetTruthValues()   
    #axCombinedFitAccuracy.scatter(numberOfPMTs, trueMeanHeight, color="orange")
    axCombinedFitAccuracy.axhline(trueMeanHeight, color="orange")
    #if fitType in ["gaussian", "twoGaussians", "Adaptive"]:
    #axCombinedFitAccuracy.errorbar(numberOfPMTs, trueMeanHeight, trueHeightSpread, color="orange")
    #else:
    #    axCombinedFitAccuracy.axhline(trueMeanHeight, color="orange")
    axCombinedFitAccuracy.set_ylabel(r"1pe Integrated Response (mV ns $\Omega$)")
    axCombinedFitAccuracy.set_xlim(0,175)
    
    getAverageHitCounts = False
    if getAverageHitCounts == True:
        averagePMTHitCounts = numpy.empty(numberOfPMTs,dtype=float)
        for fileName in fileNames:
            pmtHitCounts = GetPMTHitCounts(fileName)
            #print(pmtHitCounts)
            for i in range(numberOfPMTs):
                averagePMTHitCounts[i] += pmtHitCounts[i]
        for i in range(numberOfPMTs):
            averagePMTHitCounts[i] = averagePMTHitCounts[i] / len(fileNames)
        
        
        
        
        
        figPMTCounts, axPMTCounts = pyplot.subplots(1, 1, figsize=[32,16])
        axPMTCounts.bar(channel_numbers,averagePMTHitCounts)
        axPMTCounts.set_xlabel("PMT ID")
        axPMTCounts.set_ylabel("PMT Hit Counts")
    
    differenceFromTruth = averagePMTFittedMeans - trueMeanHeight
    print("Max difference from truth:",max(abs(differenceFromTruth)))
    