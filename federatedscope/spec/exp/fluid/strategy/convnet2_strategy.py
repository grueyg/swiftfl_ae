import copy
import random
import torch
import numpy as np

from typing import Callable, Dict, List, Optional, Tuple, cast

from federatedscope.spec.exp.fluid.strategy.base_strategy import Strategy

class ConvNet2Strategy(Strategy):
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
        # self.weight_shapes = [
        #     (32, 1, 5, 5), (32,), (64, 32, 5, 5), (64,), (2048, 3136), (2048,), (62, 2048), (62,)]
        self.weight_shapes = [tuple(para.shape) for para in self.model.parameters()]
        self.idxList = [0, 4]
        self.idxConvFC = 2

        # list to save invariant weight inidcies
        self.unchagedWeights = [[] for x in range(len(self.weight_shapes))]
        self.defDropWeights = [[] for x in range(len(self.weight_shapes))]
        self.prevDropWeights = [[] for x in range(len(self.weight_shapes))]

        # sub-model size (defualt to 0.95, will be initialized a round 2)
        self.p_val = 0.95
        self.newIteration = 5

        # update threshold (will be initialized at round 2 based on training
        # results)
        self.changeThreshold = 0.1
        self.changeIncrement = 0.01
        self.roundCounter = 0
        self.stopChange = False

        self.trainable_para_name = list(dict(list(model.named_parameters())).keys())

    def get_droppedWeights(self):
        return self.droppedWeights

    def find_stable(self,
                    weights: List,
                    results: List[Tuple[Dict,
                                        int]],
                    idxList: List[int],
                    idxConvFC: int):
        """Find the invariant neurons that are under the update threshold"""
        # Note: for each layer in the model, it has related weight parameters in 3 indices:
        #       activation, bias, and input of next layer

        # Args: parameters: global model parameters
        #          results: updated model of each client
        #          idxList: list for the starting indices of each layer
        # idxConvFC: the index of the last convolutional layer before the FC
        # layer

        difference = []
        for i in range(len(weights)):
            difference.append(torch.full(self.weight_shapes[i], 0))

        # For each client that trained on the full model, calculate which
        # weight parameters have a change below the threshold
        for cweights, num_examples, cid in results:
            clientWeights = [cweights[key] for key in self.trainable_para_name]
            if cid in self.straggler:
                continue

            for i in range(len(weights)):
                difference[i] += (torch.abs(clientWeights[i] - weights[i]) <= torch.abs(
                    self.changeThreshold * weights[i])) * 1

        # We treat weight parameter as "Invariant" only if its an "invariant"
        # weight parameter for at least 75% of the non-straggler clients
        for i in range(len(difference)):
            difference[i] = difference[i] >= (
                0.75 * (len(results) - len(self.straggler)))

        unchagedWeightsList = [[] for x in range(len(weights))]

        # (Optional) once a neuron is added to the list, keep it in there
        unchagedWeightsList[0] = self.unchagedWeights[0]

        for idx in idxList:
            dim = [x for x in range(weights[idx].ndim)]
            dim.remove(0)

            # perform reduction for all other dimensions (so we know which idx
            # has constant weights)
            idx0Layer = np.all(difference[idx].numpy(), axis=tuple(dim))
            idx1Layer = difference[idx + 1].numpy()

            dim = [x for x in range(weights[idx + 2].ndim)]
            dim.remove(1)
            idx2Layer = np.all(difference[idx + 2].numpy(), axis=tuple(dim))

            # Calculate which neuron is "invariant" for all related weight
            # parameters related to that neuron
            noChangeIdx = idx0Layer & idx1Layer & idx2Layer

            # set unchanged list
            for i in range(len(noChangeIdx)):
                if noChangeIdx[i]:
                    if i not in unchagedWeightsList[idx]:
                        unchagedWeightsList[idx].append(i)
            # print("unchanged idx ", idx, ": ", unchagedWeightsList[idx])

           # Check which neurons were dropped last round and is still in the
           # unchangedlist
            self.defDropWeights[idx] = []
            if len(self.prevDropWeights[idx]) > 0:
                for i in self.prevDropWeights[idx]:
                    if i in unchagedWeightsList[idx]:
                        self.defDropWeights[idx].append(i)
            # print("def drop idx ", idx, ": ", self.defDropWeights[idx])

        # Repeat process for last convolutional layer
        dim = [x for x in range(weights[idxConvFC].ndim)]
        dim.remove(0)
        idx0Layer = np.all(difference[idxConvFC].numpy(), axis=tuple(dim))

        idx1Layer = difference[idxConvFC + 1].numpy()

        dim = [x for x in range(weights[idxConvFC + 2].ndim)]
        dim.remove(1)
        idx2Layer = np.all(difference[idxConvFC + 2].numpy(), axis=tuple(dim))

        reducedList = np.array([])
        for i in range(len(idx0Layer)):
            # "unflatten" the first FC layer
            reducedList = np.array(np.append(reducedList, np.all(
                idx2Layer[i * 7 * 7: i * 7 * 7 + 7 * 7])), dtype=bool)

        # reducedList = torch.tensor([], dtype=torch.bool)
        # for i in range(len(idx0Layer)):
        #     # "unflatten" the first FC layer
        #     # 假设idx2Layer是一个条件张量，这里需要确保它以适当方式定义
        #     block = idx2Layer[i * 7 * 7: i * 7 * 7 + 49].view(-1, 7, 7)
        #     reducedList = torch.cat((reducedList, torch.all(block, dim=(1, 2)).unsqueeze(0)), dim=0)

        noChangeIdx = idx0Layer & idx1Layer & reducedList
        for i in range(len(noChangeIdx)):
            if noChangeIdx[i]:
                unchagedWeightsList[idxConvFC].append(i)
        # print("unchanged idx ", idxConvFC, ": ", unchagedWeightsList[idxConvFC])

        self.defDropWeights[idxConvFC] = []
        if len(self.prevDropWeights[idxConvFC]) > 0:
            for i in self.prevDropWeights[idxConvFC]:
                if i in unchagedWeightsList[idxConvFC]:
                    self.defDropWeights[idxConvFC].append(i)
        # print("def drop idx ", idxConvFC, ": ", self.defDropWeights[idxConvFC])

        self.unchagedWeights = unchagedWeightsList

        return unchagedWeightsList

    def find_min(self,
                 weights: List,
                 results: List[Tuple[Dict,
                                     int]],
                 idxList: List[int],
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
                    (torch.abs(clientWeights[i] - weights[i])) / torch.abs(weights[i])
                    )

        # For each layer, calculate the % change of each neuron based on the
        # maximum % change of its related weight parameters
        for idx in idxList:
            dim = [x for x in range(weights[idx].ndim)]
            dim.remove(0)

            # perform reduction for all other dimensions (so we know which idx
            # has constant weights
            idx0Layer = torch.amax(difference[idx], dim=tuple(dim))
            idx1Layer = difference[idx + 1]

            dim = [x for x in range(weights[idx + 2].ndim)]
            dim.remove(1)
            idx2Layer = torch.amax(difference[idx + 2], dim=tuple(dim))
            sum = torch.maximum(torch.maximum(idx0Layer, idx1Layer), idx2Layer)

            noChangeIdx = torch.argsort(sum)
            # print("% difference: ", sum[noChangeIdx[0]])

            # Take average min values of neurons as initial update Threshold
            # value
            if (idx == 0 and rnd == 2):
                self.changeThreshold = sum[noChangeIdx[0]]
                # print("threshold for ", idx, "updated to: ", self.changeThreshold)
            if (idx == 0 and rnd == 3):
                self.changeThreshold = (
                    self.changeThreshold + sum[noChangeIdx[0]]) / 2
                # print("threshold for ", idx, "updated to: ", self.changeThreshold)

    def find_stable_and_min(self, results, rnd):
        if (rnd > self.cfg.federate.total_round_num/2 and self.stopChange != True):
            self.roundCounter += 1
            if (self.roundCounter >= 10):
                self.changeThreshold += self.changeIncrement
                self.roundCounter = 0
                # print("threshold updated to: ", self.changeThreshold)
        
        model_state_dict = self.model.state_dict()
        trainable_weights = [model_state_dict[key] for key in self.trainable_para_name]
        self.find_stable(trainable_weights, results, self.idxList, self.idxConvFC)
        self.find_min(trainable_weights, results, self.idxList, rnd)

    def update_straggler(self, results, rnd):
        # sort the clients based on training duration (at round2 )
        results = [(cid, self.client_duration_info[cid-1]) for _, _, cid in results]

        if (len(self.straggler) == 0) and rnd > 1:
            def time(elem):
                return elem[1]

            results.sort(key=time)
            self.straggler[results[len(results) - 1][0]] = results[len(results) - 1][1]
            print(self.straggler)

            # Set sub-model size based on slowest client vs targert time
            stragglerDur = results[len(results) - 1][1]
            nextSlowDur = results[len(results) - 2][1]
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
            slowest = results[len(results) - 1]
            if (slowest[0] not in self.straggler):
                # estimate current straggler's original training time without
                # dropout
                for i in range(len(results)):
                    if results[i][0] in self.straggler:
                        self.straggler[results[i][0]] = results[i][1] / self.p_val
                        print("Updated estimate straggler orig time to:",
                              self.straggler[results[i][0]])
                stragglerList = list(self.straggler.items())

                # Compare slowest device against straggler's orig training time
                if (slowest[1] > stragglerList[0][1]):
                    self.straggler[slowest[0]] = slowest[1]

                    # Set sub-model size based on slowest client vs target time
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

        # self.weight_shapes = [(5, 5, 1, 16), (16,), (5, 5, 16, 64), (64,), (3136, 120), (120,), (120, 62), (62,)]
        # Weight parameters are sent back as 1D arrays from the Android, clients, it is easier to transform
        # the weights back to original multi-D shape for dropout

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
                    self.stopChange = True
                dropList = sorted(
                    random.sample(
                        self.defDropWeights[idx],
                        numToDrop))
            elif len(self.unchagedWeights[idx]) >= numToDrop:
                if (idx == 0):
                    self.stopChange = True

                fullList = self.unchagedWeights[idx].copy()
                for x in self.defDropWeights[idx]:
                    fullList.remove(x)
                dropList = random.sample(fullList,
                                     numToDrop - len(self.defDropWeights[idx]))
                dropList.extend(self.defDropWeights[idx])
                dropList.sort()
            else:
                fullList = [x for x in range(shape[first])]
                for x in self.unchagedWeights[idx]:
                    fullList.remove(x)
                dropList = random.sample(fullList,
                                     numToDrop - len(self.unchagedWeights[idx]))
                dropList.extend(self.unchagedWeights[idx])
                dropList.sort()

            # save a copy of the dropped neurons
            self.prevDropWeights[idx] = dropList.copy()
            # print("Dropped weights idx ", idx, ": ", (self.prevDropWeights[idx]))

            self.droppedWeights[cid][idx][0] = dropList.copy()
            self.droppedWeights[cid][idx + 1][0] = dropList.copy()
            self.droppedWeights[cid][idx + 2][1] = dropList.copy()

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
        if len(self.unchagedWeights[idxConvFC]) >= numToDrop:
            listFC = sorted(
                random.sample(
                    self.unchagedWeights[idxConvFC],
                    numToDrop))
        else:
            fullList = [x for x in range(shape[first])]
            for x in self.unchagedWeights[idxConvFC]:
                fullList.remove(x)
            listFC = random.sample(fullList, numToDrop -
                                   len(self.unchagedWeights[idxConvFC]))
            listFC.extend(self.unchagedWeights[idxConvFC])
            listFC.sort()
        self.prevDropWeights[idxConvFC] = listFC.copy()
        # print("Dropped weights idx ",idxConvFC,": ",(self.prevDropWeights[idxConvFC]))

        # Extend drop list for the input of first fully connected layer
        FClist = []
        for drop in listFC:
            FClist.extend(range(drop * 7 * 7, drop * 7 * 7 + 7 * 7))

        # drop the neurons from each weight parameter ( # filters for conv
        # layer, bias, # of inputs for first FC layer)

        self.droppedWeights[cid][idxConvFC][0] = listFC.copy()
        self.droppedWeights[cid][idxConvFC + 1][0] = listFC.copy()
        self.droppedWeights[cid][idxConvFC + 2][1] = FClist.copy()


        return self.droppedWeights[cid]