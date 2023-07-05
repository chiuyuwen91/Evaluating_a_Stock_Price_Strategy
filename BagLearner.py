import numpy as np
from scipy import stats

class BagLearner(object):

    def __init__(self, learner, bags, kwargs, boost, verbose):
        self.bag = bags
        self.boost = boost
        self.verbose = verbose

        self.learners = []
        for i in range(bags):
            self.learners.append(learner(**kwargs))

    def author(self):
        return "ychiu60"

    def add_evidence(self,data_x, data_y):
        rows = data_x.shape[0]
        for learner in self.learners:
            bag_index = np.random.choice(rows, rows)
            bag_x = data_x[bag_index]
            bag_y = data_y[bag_index]
            learner.add_evidence(bag_x, bag_y)

    def query(self,points):
        output = []
        for learner in self.learners:
            output.append(learner.query(points))
        output = stats.mode(np.asarray(output), axis=0)[0][0]
        return output

if __name__ == "__main__":
    print("the secret clue is 'zzyzx'")



