import uproot
import numpy
import matplotlib.pyplot as pyplot
import awkward as ak
#pyplot.rcdefaults()
pyplot.rcParams["font.size"]=32

## Copied pmt properties from PMTINFO, may want to update to use a file instead?
PMTInfoDict = {
"x":     [300.0, -300.0, 300.0, -300.0, 1025.0, -1025.0, 1025.0, -1025.0, 300.0, -300.0, 300.0, -300.0, 662.5, -662.5, 662.5, -662.5, 300.0, -300.0, 300.0, -300.0, 1025.0, -1025.0, 1025.0, -1025.0, 300.0, -300.0, 300.0, -300.0, 662.5, -662.5, 662.5, -662.5, 1487.5, 1487.5, 1487.5, 1487.5, 1242.7401679353572, 860.9025060946217, 1242.7401679353572, 860.9025060946217, 270.0000000000001, -269.9999999999999, 270.0000000000001, -269.9999999999999, -860.9025060946215, -1242.740167935357, -860.9025060946215, -1242.740167935357, -1487.5, -1487.5, -1487.5, -1487.5, -1242.7401679353575, -860.9025060946219, -1242.7401679353575, -860.9025060946219, -270.0000000000003, 269.9999999999997, -270.0000000000003, 269.9999999999997, 860.9025060946213, 1242.740167935357, 860.9025060946213, 1242.740167935357, 1487.5, 1487.5, 1487.5, 1487.5, 1242.7401679353572, 860.9025060946217, 1242.7401679353572, 860.9025060946217, 270.0000000000001, -269.9999999999999, 270.0000000000001, -269.9999999999999, -860.9025060946215, -1242.740167935357, -860.9025060946215, -1242.740167935357, -1487.5, -1487.5, -1487.5, -1487.5, -1242.7401679353575, -860.9025060946219, -1242.7401679353575, -860.9025060946219, -270.0000000000003, 269.9999999999997, -270.0000000000003, 269.9999999999997, 860.9025060946213, 1242.740167935357, 860.9025060946213, 1242.740167935357],
"y":     [300.0, 300.0, -300.0, -300.0, 300.0, 300.0, -300.0, -300.0, 1025.0, 1025.0, -1025.0, -1025.0, 662.5, 662.5, -662.5, -662.5, 300.0, 300.0, -300.0, -300.0, 300.0, 300.0, -300.0, -300.0, 1025.0, 1025.0, -1025.0, -1025.0, 662.5, 662.5, -662.5, -662.5, -270.0, 270.0, -270.0, 270.0, 860.9025060946215, 1242.740167935357, 860.9025060946215, 1242.740167935357, 1487.5, 1487.5, 1487.5, 1487.5, 1242.7401679353572, 860.9025060946217, 1242.7401679353572, 860.9025060946217, 270.00000000000017, -269.99999999999983, 270.00000000000017, -269.99999999999983, -860.9025060946213, -1242.740167935357, -860.9025060946213, -1242.740167935357, -1487.5, -1487.5, -1487.5, -1487.5, -1242.7401679353575, -860.9025060946219, -1242.7401679353575, -860.9025060946219, -270.0, 270.0, -270.0, 270.0, 860.9025060946215, 1242.740167935357, 860.9025060946215, 1242.740167935357, 1487.5, 1487.5, 1487.5, 1487.5, 1242.7401679353572, 860.9025060946217, 1242.7401679353572, 860.9025060946217, 270.00000000000017, -269.99999999999983, 270.00000000000017, -269.99999999999983, -860.9025060946213, -1242.740167935357, -860.9025060946213, -1242.740167935357, -1487.5, -1487.5, -1487.5, -1487.5, -1242.7401679353575, -860.9025060946219, -1242.7401679353575, -860.9025060946219],
"z":     [1201.0, 1201.0, 1201.0, 1201.0, 1201.0, 1201.0, 1201.0, 1201.0, 1201.0, 1201.0, 1201.0, 1201.0, 1201.0, 1201.0, 1201.0, 1201.0, -1201.0, -1201.0, -1201.0, -1201.0, -1201.0, -1201.0, -1201.0, -1201.0, -1201.0, -1201.0, -1201.0, -1201.0, -1201.0, -1201.0, -1201.0, -1201.0, -785.0, -785.0, 785.0, 785.0, -785.0, -785.0, 785.0, 785.0, -785.0, -785.0, 785.0, 785.0, -785.0, -785.0, 785.0, 785.0, -785.0, -785.0, 785.0, 785.0, -785.0, -785.0, 785.0, 785.0, -785.0, -785.0, 785.0, 785.0, -785.0, -785.0, 785.0, 785.0, -295.0, -295.0, 295.0, 295.0, -295.0, -295.0, 295.0, 295.0, -295.0, -295.0, 295.0, 295.0, -295.0, -295.0, 295.0, 295.0, -295.0, -295.0, 295.0, 295.0, -295.0, -295.0, 295.0, 295.0, -295.0, -295.0, 295.0, 295.0, -295.0, -295.0, 295.0, 295.0],
"dir_x": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0, -1.0, -0.7071067811865476, -0.7071067811865476, -0.7071067811865476, -0.7071067811865476, -6.123233995736766e-17, -6.123233995736766e-17, -6.123233995736766e-17, -6.123233995736766e-17, 0.7071067811865475, 0.7071067811865475, 0.7071067811865475, 0.7071067811865475, 1.0, 1.0, 1.0, 1.0, 0.7071067811865477, 0.7071067811865477, 0.7071067811865477, 0.7071067811865477, 1.8369701987210297e-16, 1.8369701987210297e-16, 1.8369701987210297e-16, 1.8369701987210297e-16, -0.7071067811865474, -0.7071067811865474, -0.7071067811865474, -0.7071067811865474, -1.0, -1.0, -1.0, -1.0, -0.7071067811865476, -0.7071067811865476, -0.7071067811865476, -0.7071067811865476, -6.123233995736766e-17, -6.123233995736766e-17, -6.123233995736766e-17, -6.123233995736766e-17, 0.7071067811865475, 0.7071067811865475, 0.7071067811865475, 0.7071067811865475, 1.0, 1.0, 1.0, 1.0, 0.7071067811865477, 0.7071067811865477, 0.7071067811865477, 0.7071067811865477, 1.8369701987210297e-16, 1.8369701987210297e-16, 1.8369701987210297e-16, 1.8369701987210297e-16, -0.7071067811865474, -0.7071067811865474, -0.7071067811865474, -0.7071067811865474],
"dir_y": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, -0.0, -0.0, -0.0, -0.7071067811865475, -0.7071067811865475, -0.7071067811865475, -0.7071067811865475, -1.0, -1.0, -1.0, -1.0, -0.7071067811865476, -0.7071067811865476, -0.7071067811865476, -0.7071067811865476, -1.2246467991473532e-16, -1.2246467991473532e-16, -1.2246467991473532e-16, -1.2246467991473532e-16, 0.7071067811865475, 0.7071067811865475, 0.7071067811865475, 0.7071067811865475, 1.0, 1.0, 1.0, 1.0, 0.7071067811865477, 0.7071067811865477, 0.7071067811865477, 0.7071067811865477, -0.0, -0.0, -0.0, -0.0, -0.7071067811865475, -0.7071067811865475, -0.7071067811865475, -0.7071067811865475, -1.0, -1.0, -1.0, -1.0, -0.7071067811865476, -0.7071067811865476, -0.7071067811865476, -0.7071067811865476, -1.2246467991473532e-16, -1.2246467991473532e-16, -1.2246467991473532e-16, -1.2246467991473532e-16, 0.7071067811865475, 0.7071067811865475, 0.7071067811865475, 0.7071067811865475, 1.0, 1.0, 1.0, 1.0, 0.7071067811865477, 0.7071067811865477, 0.7071067811865477, 0.7071067811865477],
"dir_z": [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
"type": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
}

print(len(PMTInfoDict["z"]))



def samplesToTime(sample):
    sampleRate = 500e6
    return sample/sampleRate

def timeToSamples(time):
    sampleRate = 500e6
    return time*sampleRate




def LoopOverEventToGetMultiHitInfo(discretisedTimes, eventPMTIDs):
    """This takes a set of the PMT IDs of each hit, and the discretised times at which they occurred.
    It then uses this to form a dictionary in which each PMT ID has a list associated with it."""
    multiHitDictionary = {  }
    uniquePMTIDs = numpy.unique(eventPMTIDs)
    for uniquePMTID in uniquePMTIDs:
        currentPMTIndexes = numpy.argwhere(eventPMTIDs == uniquePMTID) 
        currentPMTTimes = discretisedTimes[currentPMTIndexes]
        uniquePMTTimes, PMTTimesCounts = numpy.unique(currentPMTTimes, return_counts=True)
        # Remove the single-hit events
        # PMTTimesCounts = numpy.delete(PMTTimesCounts, numpy.where(PMTTimesCounts == 1))
        multiHitDictionary[uniquePMTID] = PMTTimesCounts
    
    return multiHitDictionary
        


def AddMultiHitsToCombinedDict(listToAddTo, dictToAdd):
    for pmtNumber in dictToAdd.keys():
        if len(dictToAdd[pmtNumber]) == 0:
            continue
        for multiHitsIndex in dictToAdd[pmtNumber]:
            if multiHitsIndex in listToAddTo[pmtNumber]: # If the pmt has already got that multiplicity in it, add 1 to the value
                listToAddTo[pmtNumber][multiHitsIndex] += 1
            else:
                listToAddTo[pmtNumber][multiHitsIndex] = 1
            if multiHitsIndex == 7:
                print("7 here")
    return 0




with uproot.open("CoincidenceStudy/10kEvents/coincidenceTest800Photons_lb.ntuple.root") as file: # "lightBallCalibration3_150Photons_120kEvents.ntuple.root"
    print(file.keys())
    
    #print(file["output"])
    print(file["output"].keys())
    #print(file["meta"].values())

    #runIds = file["meta"]["runId"].array()
    #print(runIds)

    #runTime = file["meta"]["macro"].array()
    #print(runTime)

    output = file["output"]
    #print("\n",file["output"]["mcxs"].array())
    #print(file["output"]["mcPMTID"].array())
    #print(file["output"]["mcPMTCharge"].array())
    
    print("There are", len(output["mcPEHitTime"].array()),"events in total.")
    
    triggerTimes = output["mcPEHitTime"].array()[0]
    print(triggerTimes)
    print("Check digitNhits:", output["digitNhits"].array())
    
    pmtIDs = output["mcPMTID"].array()[0]
    print("PMT IDs:",pmtIDs)
    pmtNHits = output["mcnhits"].array()
    print("NHits:",pmtNHits)
    
    hitpmtIDs = output["mcPMTID"].array()[0]
    testVariableName = "mcPMTCharge"#"mcPMTNPE"
    testVariable = output[testVariableName].array()
    print(testVariableName, ":", testVariable)
    
    print("Number of hits(?):",len(triggerTimes))
    print("mcPMTNPE sum:",numpy.sum(testVariable))
    
    
    #print("Charges:", output["hitPMTTime"].array()[0])
    
    fig,ax = pyplot.subplots(1, 1, figsize=[16,8])
    binEdges = numpy.arange(0,20,step=0.1)
    ax.hist(triggerTimes, bins=binEdges)
    ax.set_xlabel("Trigger Time (ns)")
    ax.set_ylabel("Count")
    #ax.axvline(5.65, color="orange")
    #ax.axvline(4.715895821501953861694546031575,color="orange")
    #ax.axvline(6.6699605251278025,color="red")
    #ax.axvline(7.557241018354886, color="red")
    
    
    figPMTHits, axPMTHits = pyplot.subplots(1, 1, figsize=[16,8])
    axPMTHits.bar(pmtIDs,testVariable[0])
    axPMTHits.set_xlabel("PMT ID")
    axPMTHits.set_ylabel(testVariableName)
    axPMTHits.set_xlim(-1,97)
    
    
    # us = numpy.array( file["output"]["mcus"].array()[0] )
    # vs = numpy.array( file["output"]["mcvs"].array()[0] )
    # ws = numpy.array( file["output"]["mcws"].array()[0] )
    # thetas = numpy.arctan2(vs,us)
    # phis = numpy.arctan2(ws,us)
    
    # figElectronVelSpread, axElectronVelSpread = pyplot.subplots(1, 1, figsize=[16,8])
    # #axElectronVelSpread.scatter(thetas,phis)
    # #axElectronVelSpread.plot(us)
    # axElectronVelSpread.plot(vs)
    # #axElectronVelSpread.plot(ws)
    # axElectronVelSpread.plot(numpy.sqrt( numpy.multiply(us,us) + numpy.multiply(vs,vs) + numpy.multiply(ws,ws)) )
    
    # print(numpy.sum(vs <= 0))
    
    
    
    
    
    vHigh = 1000 # mV
    vLow = -1000 # mV
    vOffset = 800 # mV   <- This is just a hack until I fix what the actual voltage offset should be in the simulation
    adcCountsPerMilliV = 16384 / (vHigh-vLow)
    
    # print("waveforms evID:", file["waveforms"]["evid"].array())
    # matchingIDs = numpy.where(file["waveforms"]["evid"].array()==38)
    # print("Where evID==38:", matchingIDs)
    # waveformNumber = matchingIDs[0][13]
    # print(file["waveforms"].keys())
    # print(file["waveforms"]["waveform"].array()[waveformNumber])
    # figWaveform, axWaveform = pyplot.subplots(1, 1, figsize=[10,8])
    # axWaveform.plot((file["waveforms"]["waveform"].array()[waveformNumber] / adcCountsPerMilliV + vLow - vOffset) / 1000 )
    # axWaveform.set_xlabel("Sample Number")
    # axWaveform.set_ylabel("ADC Voltage (V)")# "Digitization Value")
    # print("Waveform length:",len(file["waveforms"]["waveform"].array()[waveformNumber]))
    # secax = axWaveform.secondary_xaxis('top', functions=(samplesToTime, timeToSamples))
    # secax.set_xlabel("Time (s)")
    # print("len waveforms array:",len(file["waveforms"]["waveform"].array()))
    # print(file["waveforms"]["waveform"].array())
    # inWindowPulseTimes = file["waveforms"]["inWindowPulseTimes"].array()
    
    # print("len iWPT:",len(inWindowPulseTimes))
    # print("iWPT:",inWindowPulseTimes)
    
    
    
    
    
#     print("        Noise testing: ")
#     print(output["mcPEProcess"].array()[-2])
    
    
    
#     #for time in inWindowPulseTimes:
#     #    print(time)
#     #photonStartTimes = file["output"]["mct"].array()
#     figTimeDistribution, axTimeDistribution = pyplot.subplots(1, 1, figsize=[16,8])
#     axTimeDistribution.hist(ak.flatten(inWindowPulseTimes), bins=150)
#     axTimeDistribution.set_xlabel("Time (ns)")
#     axTimeDistribution.set_ylabel("Count")
#     axTimeDistribution.set_title("Pulse Start Time Spread")
#     print("There are",len(inWindowPulseTimes),"pulses.")
#     print("There are", len(output["mcPEHitTime"].array()),"events in total.")
# # with uproot.open("outputTestNewbutton.root") as file:
# #      print(file.keys())
# #      print(file["T"])
    
    #### Check charge mcPMTNPE distributions
    print("\n\nChecking mcPMTNPE distribution:")
    print(len(output["mcPMTNPE"].array()))
    pmtHits = numpy.zeros(96)
    for i in range(96):
        pmtHits[i] += numpy.count_nonzero( output["mcPMTID"].array() == i)
    print(pmtHits)
    print(pmtHits.sum())
    
    barEdges = numpy.arange(0,96, step=1)
    figNPEDist, axNPEDist = pyplot.subplots(1, 1, figsize=[32,16])
    axNPEDist.bar(barEdges,pmtHits)
    axNPEDist.set_xlabel("PMT ID")
    axNPEDist.set_ylabel("Number of Hits")
    


    print(numpy.sum(output["mcparticlecount"].array()),"total number of photons in")
    print(numpy.amax(pmtHits),"number of hits in most-hit pmt")
    print(numpy.amin(pmtHits),"number of hits in least-hit pmt")
    




    ### Coincidence counting
    # Want to store the number of coincidences, along with the number of hits that were involved in each coincidence
    # Say that any photons within 2ns of each other are a coincidence.
    print("\n\nStarting coincidences study:\n")
    
    # First get the list of hits, hopefully they're sorted well
    hitPEPMTIDs = (output["mcPEPMTID"].array())
    print(type(hitPEPMTIDs))
    
    # Need to get the hitPMTIDs for each dataset
    # Also need the hit times for each hit
    hitTimes = (output["mcPEHitTime"].array())
    print("len(hitTimes):",len(hitTimes))
    print("len(hitPEPMTIDs):",len(hitPEPMTIDs))
    print("hitTimes[0]",hitTimes[0])
    print("hitPEPMTIDs[0]",hitPEPMTIDs[0])
    timeSamples = numpy.arange(-80,220,step=2) # Samples
    
    # Need to loop over each event individually or else they all combine together
    
    coincidenceValues = []
    for i in range(96):
        #print(i)
        coincidenceValues.append({})
        
    #     # I've got an array of hit pmt IDs and hit PMT times with analogous structures
    #     # What I want is to produce an array that has:
    #     # 96 rows - one for each PMT
    #     # Each row consisting of each coincidence type, so e.g. 
    #     # 1 2-hit and 3 3-hits would be stored as [2,3,3,3]
    #     # Do I want to keep events separate?
    #     # I don't think I need to, since I'm going after an average number of coincidences in each case?
    #     # In that case, something like a dictionary of values with key:value pairs of "coincidenceType":"number"
        
    #     # I want to try doing it without explicitly looping over the events though because this is slow
        


    # Make a dictionary for each PMT
    multiHitList = []
    for i in range(96):
        multiHitList.append( {} )
    
    
    for eventNumber in range(len(hitTimes)):
        print(eventNumber)
        # loop over every event (a.k.a. each pulse)
        # Get the IDs of each PMT that was hit and the time of each hit in this pulse
        # These are stored in the same order (by ratpac), so if the IDs list was e.g. [4,8,33]
        # and the hit times were [0.6, 2, 5]
        # then this would mean PMT 4 was hit at time t=0.6, PMT 8 was hit at t=2 and PMT 33 was hit at t=5
        currentHitTimes = hitTimes[eventNumber]
        currentHitPEPMTIDs = hitPEPMTIDs[eventNumber]
        
        # Get the discretised times:
        # numpy searchsorted gives the index that the value would have to be inserted in to retain the order
        # E.g. with [0,1,2] as an array, if you use searchsorted with the value 0.5, it would give 1. 
        # To get the sample location, strictly speaking I want 1 less than this, but it doesn't matter as long as you're consistent
        # because you only care about whether values occur multiple times or not, not where they are exactly.
        discreteSampleLocations = numpy.searchsorted(timeSamples,currentHitTimes,side='right')-1
        # These are essentially the discretised times, but in index form.
        currentEventHitDictionary = LoopOverEventToGetMultiHitInfo(discreteSampleLocations,currentHitPEPMTIDs)
        
        #print(currentEventHitDictionary.values())
        # if 7 in currentEventHitDictionary.values():
        #     print(eventNumber,"has a 7-hit")
        #     print(currentEventHitDictionary)
        #     print(currentHitPEPMTIDs)
        #     unique7, counts7 = numpy.unique(currentHitPEPMTIDs,return_counts=True)
        #     print(unique7, counts7)
            
        # for k,v in currentEventHitDictionary.items():
        #     print(k, v)
        # print("\n")
    
        # Now that I have the dictionaries, need to combine them in the overall results:
        AddMultiHitsToCombinedDict(coincidenceValues, currentEventHitDictionary)

    print(coincidenceValues)

    #numpy.save("CoincidenceStudy_" + "800" + "_photons_10kEvents_lb", coincidenceValues)

    # I now need to combine all the values and make some kind of graph showing off the differences
    # Something like picking a pmt, or maybe the low-mid-high values and plotting them on a graph
    # of coincidences of different types vs photons in pulse
    # Looks like the lowest is always going to be zero, maybe just make examples by picking PMTs individually?
    # The max is good to know about since this is the worst PMT for calibrating
    # What a "typical" pmt looks like is hard to define because they're all hit different amounts.
    
    

    #print(coincidenceValues[0])
    