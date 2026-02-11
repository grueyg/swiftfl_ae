from federatedscope.spec.exp.oort.utils import *
import math

from random import Random
from collections import OrderedDict
import numpy as np2
import sys
import os

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def create_training_selector(args):
    return _training_selector(args)

class _training_selector(object):
    """Oort's training selector
    """
    def __init__(self, args):

        self.totalArms = OrderedDict()
        self.training_round = 0

        self.exploration = args.exploration_factor
        self.decay_factor = args.exploration_decay
        self.exploration_min = args.exploration_min
        self.alpha = args.exploration_alpha

        self.rng = Random()
        self.unexplored = set()
        self.args = args
        self.round_threshold = args.round_threshold
        self.round_prefer_duration = float('inf')
        self.last_util_record = 0

        self.sample_window = self.args.sample_window
        self.exploitUtilHistory = []
        self.exploreUtilHistory = []
        self.exploitClients = []
        self.exploreClients = []
        self.successfulClients = set()
        self.blacklist = None


    def register_client(self, clientId, feedbacks):
        # Initiate the score for arms. [score, time_stamp, # of trials, size of client, auxi, duration]
        if clientId not in self.totalArms:
            self.totalArms[clientId] = {}
            self.totalArms[clientId]['reward'] = feedbacks['reward']
            self.totalArms[clientId]['duration'] = feedbacks['duration']
            self.totalArms[clientId]['time_stamp'] = self.training_round
            self.totalArms[clientId]['count'] = 0
            self.totalArms[clientId]['status'] = True
            self.totalArms[clientId]['gradient'] = feedbacks['gradient']

            self.unexplored.add(clientId)

    def calculateSumUtil(self, clientList):
        cnt, cntUtil = 1e-4, 0

        for client in clientList:
            if client in self.successfulClients:
                cnt += 1
                cntUtil += self.totalArms[client]['reward']

        return cntUtil/cnt

    def pacer(self):
        # summarize utility in last epoch
        lastExplorationUtil = self.calculateSumUtil(self.exploreClients)
        lastExploitationUtil = self.calculateSumUtil(self.exploitClients)

        self.exploreUtilHistory.append(lastExplorationUtil)
        self.exploitUtilHistory.append(lastExploitationUtil)

        self.successfulClients = set()

        if self.training_round >= 2 * self.args.pacer_step and self.training_round % self.args.pacer_step == 0:

            utilLastPacerRounds = sum(self.exploitUtilHistory[-2*self.args.pacer_step:-self.args.pacer_step])
            utilCurrentPacerRounds = sum(self.exploitUtilHistory[-self.args.pacer_step:])

            # Cumulated statistical utility becomes flat, so we need a bump by relaxing the pacer
            if abs(utilCurrentPacerRounds - utilLastPacerRounds) <= utilLastPacerRounds * 0.1:
                self.round_threshold = min(100., self.round_threshold + self.args.pacer_delta)
                self.last_util_record = self.training_round - self.args.pacer_step
                logging.debug("Training selector: Pacer changes at {} to {}".format(self.training_round, self.round_threshold))

            # change sharply -> we decrease the pacer step
            elif abs(utilCurrentPacerRounds - utilLastPacerRounds) >= utilLastPacerRounds * 5:
                self.round_threshold = max(self.args.pacer_delta, self.round_threshold - self.args.pacer_delta)
                self.last_util_record = self.training_round - self.args.pacer_step
                logging.debug("Training selector: Pacer changes at {} to {}".format(self.training_round, self.round_threshold))

            logging.debug("Training selector: utilLastPacerRounds {}, utilCurrentPacerRounds {} in round {}"
                .format(utilLastPacerRounds, utilCurrentPacerRounds, self.training_round))

        logging.info("Training selector: Pacer {}: lastExploitationUtil {}, lastExplorationUtil {}, last_util_record {}".
                        format(self.training_round, lastExploitationUtil, lastExplorationUtil, self.last_util_record))

    def update_client_util(self, clientId, feedbacks):
        '''
        @ feedbacks['reward']: statistical utility
        @ feedbacks['duration']: system utility
        @ feedbacks['count']: times of involved
        '''
        self.totalArms[clientId]['reward'] = feedbacks['reward']
        self.totalArms[clientId]['duration'] = feedbacks['duration']
        self.totalArms[clientId]['time_stamp'] = feedbacks['time_stamp']
        self.totalArms[clientId]['count'] += 1
        self.totalArms[clientId]['status'] = feedbacks['status']
        self.totalArms[clientId]['gradient'] = feedbacks['gradient']

        self.unexplored.discard(clientId)
        self.successfulClients.add(clientId)

    def get_blacklist(self):
        blacklist = []

        if self.args.blacklist_rounds != -1:
            sorted_client_ids = sorted(list(self.totalArms), reverse=True,
                                        key=lambda k:self.totalArms[k]['count'])

            for clientId in sorted_client_ids:
                if self.totalArms[clientId]['count'] > self.args.blacklist_rounds:
                    blacklist.append(clientId)
                else:
                    break

            # we need to back up if we have blacklisted all clients
            predefined_max_len = self.args.blacklist_max_len * len(self.totalArms)

            if len(blacklist) > predefined_max_len:
                logging.warning("Training Selector: exceeds the blacklist threshold")
                blacklist = blacklist[:predefined_max_len]

        return set(blacklist)

    def select_participant(self, num_of_clients, feasible_clients=None):
        '''
        @ num_of_clients: # of clients selected
        '''
        viable_clients = feasible_clients if feasible_clients is not None else set([x for x in self.totalArms.keys() if self.totalArms[x]['status']])
        return self.getTopK(num_of_clients, self.training_round+1, viable_clients)

    def update_duration(self, clientId, duration):
        if clientId in self.totalArms:
            self.totalArms[clientId]['duration'] = duration

    def getTopK(self, numOfSamples, cur_time, feasible_clients):
        self.training_round = cur_time
        self.blacklist = self.get_blacklist()

        self.pacer()
        # try: 
        # normalize the score of all arms: Avg + Confidence
        scores = {}
        numOfExploited = 0
        exploreLen = 0

        client_list = list(self.totalArms.keys())
        orderedKeys = [x for x in client_list if int(x) in feasible_clients and int(x) not in self.blacklist]


        if self.round_threshold < 100.:
            sortedDuration = sorted([self.totalArms[key]['duration'] for key in client_list])
            self.round_prefer_duration = sortedDuration[min(int(len(sortedDuration) * self.round_threshold/100.), len(sortedDuration)-1)]
        else:
            self.round_prefer_duration = float('inf')

        moving_reward, staleness, allloss = [], [], {}

        for clientId in orderedKeys:
            if self.totalArms[clientId]['reward'] > 0:
                creward = self.totalArms[clientId]['reward']
                moving_reward.append(creward)
                staleness.append(cur_time - self.totalArms[clientId]['time_stamp'])


        max_reward, min_reward, range_reward, avg_reward, clip_value = self.get_norm(moving_reward, self.args.clip_bound)
        max_staleness, min_staleness, range_staleness, avg_staleness, _ = self.get_norm(staleness, thres=1)

        for key in orderedKeys:
            # we have played this arm before
            if self.totalArms[key]['count'] > 0:
                creward = min(self.totalArms[key]['reward'], clip_value)
                numOfExploited += 1

                sc = (creward - min_reward)/float(range_reward) \
                    + math.sqrt(0.1*math.log(cur_time)/self.totalArms[key]['time_stamp']) # temporal uncertainty

                #sc = (creward - min_reward)/float(range_reward) \
                #    + self.alpha*((cur_time-self.totalArms[key]['time_stamp']) - min_staleness)/float(range_staleness)

                clientDuration = self.totalArms[key]['duration']
                if clientDuration > self.round_prefer_duration:
                    sc *= ((float(self.round_prefer_duration)/max(1e-4, clientDuration)) ** self.args.round_penalty)

                if self.totalArms[key]['time_stamp']==cur_time:
                    allloss[key] = sc

                scores[key] = sc

        #ratio
        clientLakes = list(scores.keys())
        self.exploration = max(self.exploration*self.decay_factor, self.exploration_min)
        exploitLen = min(int(numOfSamples*(1.0 - self.exploration)), len(clientLakes)-1)

        # take the top-k, and then sample by probability, take 95% of the cut-off loss
        sortedClientUtil = sorted(scores, key=scores.get, reverse=True)
        logging.info("scores are {},{},{}".format(exploitLen,scores,sortedClientUtil))
        # take cut-off utility
        cut_off_util = scores[sortedClientUtil[exploitLen]] * self.args.cut_off_util

        pickedClients = []
        for clientId in sortedClientUtil:
            if scores[clientId] < cut_off_util:
                break
            pickedClients.append(clientId)

        augment_factor = len(pickedClients)

        # totalSc = max(1e-4, float(sum([scores[key] for key in pickedClients])))
        # totalSc = (sum([scores[key] for key in pickedClients]))
        poss_sample=[scores[key] for key in pickedClients]
        poss_sample=np2.asarray(poss_sample).astype('float64')
        poss_sample = poss_sample / np2.sum(poss_sample)

        pickedClients = list(np2.random.choice(pickedClients, exploitLen, p=poss_sample, replace=False))
        self.exploitClients = pickedClients

        logging.info("exploitation poss sum is {}".format(np2.sum(poss_sample)))

        # exploration
        _unexplored = [x for x in list(self.unexplored) if int(x) in feasible_clients]
        if len(_unexplored) > 0:
            init_reward = {}
            for cl in _unexplored:
                init_reward[cl] = self.totalArms[cl]['reward']
                clientDuration = self.totalArms[cl]['duration']

                if clientDuration > self.round_prefer_duration:
                    init_reward[cl] *= ((float(self.round_prefer_duration)/max(1e-4, clientDuration)) ** self.args.round_penalty)

            # prioritize w/ some rewards (i.e., size)
            exploreLen = min(len(_unexplored), numOfSamples - len(pickedClients))
            pickedUnexploredClients = sorted(init_reward, key=init_reward.get, reverse=True)[:min(int(self.sample_window*exploreLen), len(init_reward))]

            # unexploredSc = float(sum([init_reward[key] for key in pickedUnexploredClients]))
            # unexploredSc = (sum([init_reward[key] for key in pickedUnexploredClients]))
            poss_sample=[init_reward[key] for key in pickedUnexploredClients]
            poss_sample=np2.asarray(poss_sample).astype('float64')
            logging.info("exploration poss is {}"
            .format(poss_sample))
            poss_sample = poss_sample / np2.sum(poss_sample)
            # pickedUnexplored = list(np2.random.choice(pickedUnexploredClients, exploreLen, p=[init_reward[key]/max(1e-4, unexploredSc) for key in pickedUnexploredClients], replace=False))
            
            if np2.sum(poss_sample)>1e-1:
                pickedUnexplored = list(np2.random.choice(pickedUnexploredClients, exploreLen, p=poss_sample, replace=False))

                self.exploreClients = pickedUnexplored
                pickedClients = pickedClients + pickedUnexplored
            else:
                # no clients left for exploration
                pickedUnexplored = list(np2.random.choice(pickedUnexploredClients, exploreLen, replace=False))

                self.exploreClients = pickedUnexplored
                pickedClients = pickedClients + pickedUnexplored
        else:
            # no clients left for exploration
            self.exploration_min = 0.
            self.exploration = 0.

        while len(pickedClients) < numOfSamples:
            nextId = self.rng.choice(orderedKeys)
            if nextId not in pickedClients:
                pickedClients.append(nextId)

        top_k_score = []
        for i in range(min(3, len(pickedClients))):
            clientId = pickedClients[i]
            _score = (self.totalArms[clientId]['reward'] - min_reward)/range_reward
            _staleness = self.alpha*((cur_time-self.totalArms[clientId]['time_stamp']) - min_staleness)/float(range_staleness) #math.sqrt(0.1*math.log(cur_time)/max(1e-4, self.totalArms[clientId]['time_stamp']))
            top_k_score.append((self.totalArms[clientId], [_score, _staleness]))

        logging.info("At round {}, UCB exploited {}, augment_factor {}, exploreLen {}, un-explored {}, exploration {}, round_threshold {}, sampled score is {}"
            .format(cur_time, numOfExploited, augment_factor/max(1e-4, exploitLen), exploreLen, len(self.unexplored), self.exploration, self.round_threshold, top_k_score))
        # logging.info("At time {}, all rewards are {}".format(cur_time, allloss))
        # except Exception as e:
        #         exc_type, exc_obj, exc_tb = sys.exc_info()
        #         fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        #         print("====Error: " + str(e) + '\n')
        #         logging.info("====Error: {}, {}, {}, {}".format(e, exc_type, fname, exc_tb.tb_lineno))

        return pickedClients

    def get_median_reward(self):
        feasible_rewards = [self.totalArms[x]['reward'] for x in list(self.totalArms.keys()) if int(x) not in self.blacklist]

        # we report mean instead of median
        if len(feasible_rewards) > 0:
            return sum(feasible_rewards)/float(len(feasible_rewards))

        return 0

    def get_client_metric(self, armId):
        return self.totalArms[armId]

    def getAllMetrics(self):
        return self.totalArms

    def get_norm(self, aList, clip_bound=0.95, thres=1e-4):
        aList.sort()
        clip_value = aList[min(int(len(aList)*clip_bound), len(aList)-1)]

        _max = max(aList)
        _min = min(aList)*0.999
        _range = max(_max - _min, thres)
        _avg = sum(aList)/max(1e-4, float(len(aList)))

        return float(_max), float(_min), float(_range), float(_avg), float(clip_value)
