import numpy
import matplotlib.pyplot as pyplot
import awkward as ak
pyplot.rcParams["font.size"]=32

pulsePhotonNumbers = [50,100,150,200,250,300,350,400,450,500,550,600,800]

pmtNumber = 72 # 72 

dataToPlot = {}

multiHitMultiplicities = numpy.array([])
for i in range(len(pulsePhotonNumbers)):
    coincidenceValues = numpy.load("CoincidenceStudy_" + str(pulsePhotonNumbers[i]) + "_photons_10kEvents.npy", allow_pickle=True)
    keysInThisDataset = coincidenceValues[pmtNumber].keys()
    multiHitMultiplicities = numpy.append(multiHitMultiplicities, list(keysInThisDataset))
    multiHitMultiplicities = numpy.unique(multiHitMultiplicities)

print(multiHitMultiplicities)

# give each multiplicity an empty list to fill with event counts.
for multiplicity in multiHitMultiplicities:
    dataToPlot[multiplicity] = []

for i in range(len(pulsePhotonNumbers)):
    coincidenceValues = numpy.load("CoincidenceStudy_" + str(pulsePhotonNumbers[i]) + "_photons_10kEvents.npy", allow_pickle=True)

    
    # for k in coincidenceValues[pmtNumber].keys():
    #     print(k)
    #     if k not in dataToPlot:
    #         dataToPlot[k] = []
    #     dataToPlot[k].append(coincidenceValues[pmtNumber][k])
    
    for multiplicity in multiHitMultiplicities:
        if multiplicity in coincidenceValues[pmtNumber].keys():
            dataToPlot[multiplicity].append(coincidenceValues[pmtNumber][multiplicity])
        else:
            dataToPlot[multiplicity].append(0)


#print(dataToPlot)

figSinglePMTCoincidences, axSinglePMTCoincidences = pyplot.subplots(1, 1, figsize=[32,16])
for k in dataToPlot.keys():
    axSinglePMTCoincidences.plot(pulsePhotonNumbers, numpy.array(dataToPlot[k]) / 10000, label=k)
    
axSinglePMTCoincidences.legend(title="Hit Multiplicity")
axSinglePMTCoincidences.set_xlabel("Number of Photons in Pulse")
axSinglePMTCoincidences.set_ylabel("Number of Occurences per Pulse (Average)")


    
    # Issue is that some of the higher values don't have 0s where they haven't appeared before
    # Could get the unique values beforehand and then loop through pulsePhotonNumbers,
    # that way I could assign zeroes if the key doesn't exist