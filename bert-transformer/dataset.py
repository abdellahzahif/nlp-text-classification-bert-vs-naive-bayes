from torch.utils.data import Dataset
import sys
import json
import config as CONFIG




class RecipERTDataset(Dataset):

    def __init__(self, class2int, recipes):
        super(RecipERTDataset, self).__init__()
        self.class2int = class2int
        self.recipes = recipes
        # FIXME: more attribuets?

    def __getitem__(self, index):
        '''
            returns a recipe from the dataset, as a dictionary of the following form:

            {
              'tokens': tokens [0, 17, 43, 563, 372, ...]
              'labels': ...
            }

            where
            - tokens is a sequence of token IDs produced by the BERT tokenizer, and
            - labels is a boolean torch tensor with as many entries as there are recipe classes
                     (each entry indicates if the recipe belongs to a certain category).
        '''
        # FIXME
        return {'tokens': tokens, 'labels': labels}

    def __len__(self):
        return len(self.recipes)



def train_vocabulary_of_recipe_classes(recipes, nclasses):
    '''
        given a collection of recipes, identify the most frequent recipe classes.
        return a dictionary mapping each class to an integer ID, of the following form:

        class2int = {
          'Hauptspeise': 0,
          'Überbacken': 1,
          ...
        }
    '''
    # FIXME: compute
    class2int = None
    return class2int


if __name__ == '__main__':
    with open(sys.argv[1], 'r') as f:
        recipes = json.load(f)
    class2int = train_vocabulary_of_recipe_classes(recipes, CONFIG.NCLASSES)
    dataset = RecipERTDataset(class2int, recipes)
    print("============================================")
    print( dataset[0] )
    print("============================================")
    print( dataset[1] )
