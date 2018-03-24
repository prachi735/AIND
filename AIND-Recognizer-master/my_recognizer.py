import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    # return probabilities, guesses
    for word_id in range(len(test_set.get_all_Xlengths())):
        
        current_sequence, current_length = test_set.get_item_Xlengths(word_id)
        probability_dict = {}
       
        for word, model in models.items():
            try:
                probability_dict[word] = model.score(current_sequence, current_length)
            except:
                probability_dict[word] = float('-inf')
                continue

        probabilities.append(probability_dict)
        best_score = max(probability_dict, key = probability_dict.get)
        guesses.append(best_score)

    return probabilities, guesses
