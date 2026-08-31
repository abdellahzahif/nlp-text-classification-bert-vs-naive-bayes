
import sys
import json
import pickle


def apply_model(model, class2int, data_test):

    # FIXME
    # load test data
    # load class2int
    # for each test recipe:
    #   apply the model, obtaining scores
    #   threshold the scores at 40%, obtaining a set of predicted classes
    #   measure f1 between the estimated and true classes
    # report the f1, averaged over all test recipes.
    pass


if __name__ == '__main__':

    if len(sys.argv) < 3:
        print('USAGE: print_sample.py <model_checkpoint> <class2int.pkl> <test_path>')
        exit(1)

    with open(sys.argv[3], 'r') as f:
        data_test = json.load(f)
    with open(sys.argv[2], 'rb') as f:
        class2int = pickle.load(f)

    # FIXME: load model
    model = None

    apply_model(model, class2int, data_test)

