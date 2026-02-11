import copy
import random
import torch
import numpy as np

from typing import Callable, Dict, List, Optional, Tuple, cast

from federatedscope.spec.exp.fluid.strategy.base_strategy import Strategy

class ResNet18Strategy(Strategy):
    def __init__(self, client_duration_info, config, model):
        super().__init__(client_duration_info, config)

        self.model = model
        self.droppedWeights: Dict[str, List] = {}
        # list of straggler cids
        self.straggler: Dict[str, float] = {}
        self.JustUpdatedStrag = False

        # Multi dimension shape of the weight parameters
        # all shapes are 1-d when sent back from Android clients, hence we need
        # to reshape it when sent back to server
        self.weight_shapes = [(64, 3, 7, 7), (64,), (64,),  # 0
                              
                              (64, 64, 3, 3), (64,), (64,), # 3
                              (64, 64, 3, 3), (64,), (64,), # 6

                              (64, 64, 3, 3), (64,), (64,), # 9
                              (64, 64, 3, 3), (64,), (64,), # 12

                              (128, 64, 3, 3), (128,), (128,), # 15
                              (128, 128, 3, 3), (128,), (128,), # 18
                              (128, 64, 1, 1), (128,), (128,), # 21

                              (128, 128, 3, 3), (128,), (128,), # 24
                              (128, 128, 3, 3), (128,), (128,), # 27

                              (256, 128, 3, 3), (256,), (256,), # 30
                              (256, 256, 3, 3), (256,), (256,), # 33
                              (256, 128, 1, 1), (256,), (256,), # 36

                              (256, 256, 3, 3), (256,), (256,), # 39
                              (256, 256, 3, 3), (256,), (256,), # 42

                              (512, 256, 3, 3), (512,), (512,), # 45
                              (512, 512, 3, 3), (512,), (512,), # 48
                              (512, 256, 1, 1), (512,), (512,), # 51
                              
                              (512, 512, 3, 3), (512,), (512,), # 54
                              (512, 512, 3, 3), (512,), (512,), # 57
                              (10, 512), # 60
                              (10,)] # 63

        self.weight_shapes = [tuple(para.shape) for para in self.model.parameters()]

        self.idxDict = {
            3: [0],
            6: [3],
            9: [(0,6)],
            12: [9],
            15: [((0, 6), 12)],
            18: [15],
            21: [],
            24: [(18,)]
        }

        self.idxList_3 = [0, 3, 6, 9, 15, 18, 24, 27, 30, 33] # +3
        self.idxList_6 = [21, 36]
        self.idxList_9 = [3, 12, ] # +9
        self.idxConvFC = 10

        self.unchagedWeights = [[] for x in range(len(self.weight_shapes))]
        self.defDropWeights = [[] for x in range(len(self.weight_shapes))]
        self.prevDropWeights = [[] for x in range(len(self.weight_shapes))]

        # sub-model size (defualt to 0.95, will be initialized a round 2)
        self.p_val = 0.95
        self.newIteration = 5
        self.append = False

        # update threshold (will be initialized at round 2 based on training results)
        # FOR CIFAR10 since all layer have drastically different weight update % pattern,
        # we assign an individual threshold for each layer
        self.changeThreshold = [1.5 for x in range(len(self.weight_shapes))]
        self.changeIncrement = 0.2
        self.roundCounter = 0
        self.stopChange = [False for x in range(len(self.weight_shapes))]

        self.trainable_para_name = list(dict(list(model.named_parameters())).keys())


    def drop_dynamic(
            self,
            weights: Dict,
            cid: str):
        """Invariant Dropout - create sub-models based on unchanging neurons """
        # Note: for each layer in the model, it has related weight parameters in 3 indices:
        #       activation, bias, and input of next layer

        # Args: parameters: global model parameters
        #          p: sub-model size
        #          idxList: list for the starting indices of each layer
        #          idxConvFC: the index of the last convolutional layer before the FC layer
        #          cid: the id of the straggler client

        weights = list(copy.deepcopy(weights).values())
        idxConvFC = self.idxConvFC

        # Initialize list variables for the straggler
        if cid not in self.droppedWeights:
            self.droppedWeights[cid] = [[[], []] for x in range(len(weights))]

        # for each layer, select (1-p)% neurons to dropout
        for idx in self.idxList:
            # indices of the corresponding dimension that the weight matrix is reduced
            # ie for convolutional layers it is the number of filters, for FC
            # layers its the number of output neurons
            first = 0
            second = 0
            third = 1

            # calculate number of neurons to drop based on shape
            shape = weights[idx].shape
            numToDrop = shape[first] - int(self.p_val * shape[first])

            # first, prioritize dropping any neurons in the defDropWeights list (repeatedly under threshold)
            # next drop neurons that are under the threshold for this round
            # Finally, randomly drop neurons if needed
            if len(self.defDropWeights[idx]) >= numToDrop:
                if (idx == 0):
                    self.stopChange[idx] = True
                dropList = sorted(
                    random.sample(
                        self.defDropWeights[idx],
                        numToDrop))
            elif len(self.unchagedWeights[idx]) >= numToDrop:
                self.stopChange[idx] = True

                fullList = self.unchagedWeights[idx].copy()
                for x in self.defDropWeights[idx]:
                    fullList.remove(x)
                dropList = random.sample(
                    fullList, numToDrop - len(self.defDropWeights[idx]))
                dropList.extend(self.defDropWeights[idx])
                dropList.sort()

            else:
                fullList = [x for x in range(shape[first])]
                for x in self.unchagedWeights[idx]:
                    fullList.remove(x)
                dropList = random.sample(
                    fullList, numToDrop - len(self.unchagedWeights[idx]))
                dropList.extend(self.unchagedWeights[idx])
                dropList.sort()

            # save a copy of the dropped neurons
            self.prevDropWeights[idx] = dropList.copy()
            # print("Dropped weights idx ", idx, ": ", (self.prevDropWeights[idx]))

            self.droppedWeights[cid][idx][0] = dropList.copy()
            self.droppedWeights[cid][idx + 3][1] = dropList.copy()

        # Dropping for the last convolutional layer (has to be flattened for FC
        # layer)

        # indices of the corresponding dimension that the weight matrix is reduced (
        # ie for convolutional layers it is the number of filters, for FC
        # layers its the number of output neurons
        first = 0
        second = 0
        third = 1

        shape = weights[idxConvFC].shape

        numToDrop = shape[first] - int(self.p_val * shape[first])
        if len(self.unchagedWeights[self.idxConvFC]) >= numToDrop:
            listFC = sorted(
                random.sample(
                    self.unchagedWeights[self.idxConvFC],
                    numToDrop))
        else:
            fullList = [x for x in range(shape[first])]
            for x in self.unchagedWeights[self.idxConvFC]:
                fullList.remove(x)
            listFC = random.sample(fullList, numToDrop -
                                   len(self.unchagedWeights[self.idxConvFC]))
            listFC.extend(self.unchagedWeights[self.idxConvFC])
            listFC.sort()
        self.prevDropWeights[idxConvFC] = listFC.copy()
        # print("Dropped weights idx ", idxConvFC, ": ", (self.prevDropWeights[idxConvFC]))

        # Extend drop list for the input of first fully connected layer
        FClist = []
        for drop in listFC:
            FClist.extend(range(drop * 4 * 4, drop * 4 * 4 + 4 * 4))

        # drop the neurons from each weight parameter ( # filters for conv
        # layer, bias, # of inputs for first FC layer)
        self.droppedWeights[cid][idxConvFC][0] = listFC.copy()
        self.droppedWeights[cid][idxConvFC + 1][0] = listFC.copy()
        self.droppedWeights[cid][idxConvFC + 2][1] = FClist.copy()

        return self.droppedWeights[cid]

    def find_stable(self,
                    weights: List,
                    results: List[Tuple[Tuple, int]],
                    rnd=None):
        """Find the invariant neurons that are under the update threshold"""
        # Note: for each layer in the model, it has related weight parameters in 3 indices:
        #       activation, bias, and input of next layer

        # Args: parameters: global model parameters
        #          results: updated model of each client
        #          idxList: list for the starting indices of each layer
        # idxConvFC: the index of the last convolutional layer before the FC
        # layer

        # (Optional) once a neuron is added to the list, keep it in there
        unchagedWeightsList = [[] for x in range(len(weights))]

        # Since each layer has its own threshold, we check if parameters fall
        # under the threshold individually
        for idx in self.idxList:
            difference = []
            for i in range(len(weights)):
                difference.append(torch.full(self.weight_shapes[i], 0))

            for clientWeights, num_examples, cid in results:
                if cid in self.straggler:
                    continue

                for i in range(len(weights)):
                    difference[i] += (torch.abs(clientWeights[i] - weights[i]) <= torch.abs(
                        self.changeThreshold[idx] * weights[i])) * 1

            # We treat weight parameter as "Invariant" only if its an
            # "invariant" weight parameter for at least 75% of the
            # non-straggler clients
            for i in range(len(difference)):
                difference[i] = difference[i] >= (
                    0.75 * (len(results) - len(self.straggler)))

            dim = [x for x in range(weights[idx].ndim)]
            dim.remove(0)

            # perform reduction for all other dimensions (so we know which idx
            # has constant weights
            idx0Layer = np.all(difference[idx].numpy(), axis=tuple(dim))
            idx1Layer = difference[idx + 1].numpy()

            dim = [x for x in range(weights[idx + 2].ndim)]
            dim.remove(1)
            idx2Layer = torch.all(difference[idx + 2].numpy(), axis=tuple(dim))

            # Calculate which neuron is "invariant" for all related weight
            # parameters related to that neuron
            noChangeIdx = idx0Layer & idx1Layer & idx2Layer

            # set unchanged list
            for i in range(len(noChangeIdx)):
                if noChangeIdx[i]:
                    if i not in unchagedWeightsList[idx]:
                        unchagedWeightsList[idx].append(i)
            # print("unchanged idx ", idx, ": ", unchagedWeightsList[idx])

            # Check whcih neurons were dropped last round and is still in the
            # unchangedlist
            self.defDropWeights[idx] = []
            if len(self.prevDropWeights[idx]) > 0:
                for i in self.prevDropWeights[idx]:
                    if i in unchagedWeightsList[idx]:
                        self.defDropWeights[idx].append(i)
            # print("def drop idx ", idx, ": ", self.defDropWeights[idx])

        # Repeat process for last convolutional layer
        dim = [x for x in range(weights[self.idxConvFC].ndim)]
        dim.remove(0)
        idx0Layer = np.all(difference[self.idxConvFC], axis=tuple(dim))

        idx1Layer = difference[self.idxConvFC + 1]

        dim = [x for x in range(weights[self.idxConvFC + 2].ndim)]
        dim.remove(1)
        idx2Layer = np.all(difference[self.idxConvFC + 2], axis=tuple(dim))

        reducedList = np.array([])
        for i in range(len(idx0Layer)):
            # "unflatten" the first FC layer
            reducedList = np.array(np.append(reducedList, torch.all(
                idx2Layer[i * 4 * 4: i * 4 * 4 + 4 * 4])), dtype=bool)

        noChangeIdx = idx0Layer & idx1Layer & reducedList

        for i in range(len(noChangeIdx)):
            if noChangeIdx[i]:
                unchagedWeightsList[self.idxConvFC].append(i)
        # print("unchanged idx ", self.idxConvFC, ": ", list[self.idxConvFC])

        self.defDropWeights[self.idxConvFC] = []
        if len(self.prevDropWeights[self.idxConvFC]) > 0:
            for i in self.prevDropWeights[self.idxConvFC]:
                if i in unchagedWeightsList[self.idxConvFC]:
                    self.defDropWeights[self.idxConvFC].append(i)
        # print("def drop idx ", idxConvFC, ": ", self.defDropWeights[idxConvFC])

        self.unchagedWeights = unchagedWeightsList

        return unchagedWeightsList

    def find_min(self,
                 weights: List,
                 results: List[Tuple[Tuple, int]],
                 rnd: int):
        """Find the Minimum percent change for each layer of the model this round"""
        # Note: for each layer in the model, it has related weight parameters in 3 indices:
        #       activation, bias, and input of next layer

        # Args: parameters: global model parameters
        #          results: updated model of each client
        #          idxList: list for the starting indices of each layer
        #          rnd: Current round of training

        difference = []
        for i in range(len(weights)):
            difference.append(torch.full(self.weight_shapes[i], 0.0))

        # Calculate the maximum change of each weight parameter for all clients
        for cweights, num_examples, cid in results:
            clientWeights = [cweights[key] for key in self.trainable_para_name]
            if cid in self.straggler:
                continue

            for i in range(len(weights)):
                difference[i] = torch.maximum(
                    difference[i],
                    (torch.abs(
                        clientWeights[i] -
                        weights[i])) /
                    torch.abs(
                        weights[i]))

        # For each layer, calculate the % change of each neuron based on the
        # maximum % change of its related weight parameters
        for idx in self.idxList:
            dim = [x for x in range(weights[idx].ndim)]
            dim.remove(weights[idx].ndim - 1)

            # perform reduction for all other dimensions (so we know which idx
            # has constant weights
            idx0Layer = torch.amax(difference[idx], axis=tuple(dim))

            idx1Layer = difference[idx + 1]

            dim = [x for x in range(weights[idx + 2].ndim)]
            dim.remove(weights[idx + 2].ndim - 2)
            idx2Layer = torch.amax(difference[idx + 2], axis=tuple(dim))
            sum = torch.maximum(torch.maximum(idx0Layer, idx1Layer), idx2Layer)

            noChangeIdx = torch.argsort(sum)

            print("% difference: ", sum[noChangeIdx[0]])

            # Take average min values of neurons as initial update Threshold
            # value
            if (rnd == 2):
                self.changeThreshold[idx] = sum[noChangeIdx[0]]
                # print("threshold for ",idx,"updated to: ",self.changeThreshold[idx])
            if (rnd == 3):
                self.changeThreshold[idx] = (
                    self.changeThreshold[idx] + sum[noChangeIdx[0]]) / 2
                # print("threshold for ",idx,"updated to: ",self.changeThreshold[idx])

    def find_stable_and_min(self, results, rnd):
        if (rnd > self.cfg.federate.total_round_num / 2):
            self.roundCounter += 1
            if (self.roundCounter >= 2):
                for idx in self.idxList:
                    if not self.stopChange[idx]:
                        self.changeThreshold[idx] += self.changeIncrement
                        self.roundCounter = 0
                        # print(
                        #     "threshold for",
                        #     idx,
                        #     "updated to: ",
                        #     self.changeThreshold[idx])
                        
        self.find_stable(self.model.state_dict(), results, self.idxList, self.idxConvFC)
        self.find_min(self.model.state_dict(), results, self.idxList)

    def update_straggler(self, results, rnd):

        results = [(cid, self.client_duration_info[cid-1]) for _, _, cid in results]

        # sort the clients based on training duration (at round2 )
        if (len(self.straggler) == 0) and rnd > 0:
            def time(elem):
                return elem[1]

            results.sort(key=time)
            self.straggler[results[-1][0]] = results[-1][1]

            # Set sub-model size based on slowest client vs targert time
            stragglerDur = results[-1][1]
            nextSlowDur = results[-2][1]
            percentDiff = nextSlowDur / stragglerDur
            if (percentDiff >= 0.90):
                self.p_val = 0.95
            elif (percentDiff < 0.90 and percentDiff >= 0.80):
                self.p_val = 0.85
            elif (percentDiff < 0.80 and percentDiff >= 0.70):
                self.p_val = 0.75
            elif (percentDiff < 0.70 and percentDiff >= 0.60):
                self.p_val = 0.65
            else:
                self.p_val = 0.5
            # print("Updated p val to:", self.p_val)

        # for remaining rounds check if straggler changed
        elif (len(self.straggler) != 0) and rnd > 1 and self.JustUpdatedStrag == False:
            def time(elem):
                return elem[1]

            results.sort(key=time)
            slowest = results[-1]
            if (slowest[0] not in self.straggler):
                # estimate current straggler's original training time without dropout
                for i in range(len(results)):
                    if results[i][0] in self.straggler:
                        self.straggler[results[i][0]] = (results[i][1] / self.p_val) * 1.15
                        # print("Updated estimate straggler orig time to:",self.straggler[results[i][0]])
                stragglerList = list(self.straggler.items())

                # Compare slowest device against straggler's orig training time
                if (slowest[1] > stragglerList[0][1]):
                    self.straggler[slowest[0]] = slowest[1]

                    # Set sub-model size based on slowest client vs targert
                    # time
                    stragglerDur = slowest[1]
                    nextSlowDur = stragglerList[0][1]
                    percentDiff = nextSlowDur / stragglerDur
                    if (percentDiff >= 0.90):
                        self.p_val = 0.95
                    elif (percentDiff < 0.90 and percentDiff >= 0.80):
                        self.p_val = 0.85
                    elif (percentDiff < 0.80 and percentDiff >= 0.70):
                        self.p_val = 0.75
                    elif (percentDiff < 0.70 and percentDiff >= 0.60):
                        self.p_val = 0.65
                    else:
                        self.p_val = 0.5
                    # print("Updated p val to:", self.p_val)
                    self.JustUpdatedStrag = True

                    self.straggler.pop(stragglerList[0][0])
                    self.droppedWeights.pop(stragglerList[0][0])
                    stragglerList.pop(0)
            # print(self.straggler)
        else:
            self.JustUpdatedStrag = False

        return self.straggler