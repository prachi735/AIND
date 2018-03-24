import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        best_bic_score = np.float('Inf') 
        best_model = None
        
        for n in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = self.base_model(n)

                model_likelihood = model.score(self.X, self.lengths)  
                
                features = self.X.shape[1]
                p = n ** 2 + (2 * n * features) - 1
                bic_score = (- 2 * model_likelihood) + (p * np.log(self.X.shape[0]))
                
                # The lower the BIC value/score, the better the model
                if bic_score < best_bic_score:
                    best_bic_score = bic_score  
                    best_model = self.base_model(n)                   
            except:
                pass
            
        return best_model




class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        best_dic_score = np.float('-inf')
        best_n = self.min_n_components
        
        for n in range(self.min_n_components, self.max_n_components+1):
            model = self.base_model(n)
            try:
                # log likelihood of data belonging to hmm model
                current_word_in_data = model.score(self.X, self.lengths)
                # anti log likelihood = likelihood of data belonging to competing words
                avg_anti_data = 0  
                count_other_words = 0
                for word in self.words:
                    if word == self.this_word: 
                        continue
                    other_X, other_lengths = self.hwords[word]
                    avg_anti_data += model.score(other_X, other_lengths)
                    count_other_words += 1
                
                    avg_anti_data /= count_other_words
                    dic_score = current_word_in_data - avg_anti_data
                
                    if dic_score >= best_dic_score:
                        best_dic_score = dic_score
                        best_n = n
            except:
                pass

        return self.base_model(best_n)



class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        if len(self.sequences) == 1:
            return self.base_model(self.n_constant)
            
        best_model = None
        best_splits_likelihood = np.float('-inf')
        n_splits = 3 if len(self.sequences) >= 3 else len(self.sequences)
        split_method = KFold(n_splits)
        validation_scores = [] 
        best_n = self.min_n_components
        
        for n in range(self.min_n_components, self.max_n_components+1):
            
            scores = []
            
            for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                
                self.X, self.lengths = combine_sequences(cv_train_idx, self.sequences)
                test_X, test_lengths = combine_sequences(cv_test_idx, self.sequences)
                
                try:
                    model = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000,
                                            random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                    validation_scores = model.score(test_X, test_lengths)
                    validation_scores.append(scores)
                except:
                    pass
                     
            splits_likelihood = np.average(scores) if len(scores) > 0 else float("-inf")
            
            # Update best log likelihood and best n
            if splits_likelihood > best_splits_likelihood:
                best_splits_likelihood = splits_likelihood
                best_n = n
        
        best_model = GaussianHMM(n_components=best_n, covariance_type="diag", n_iter=1000,
                                 random_state=self.random_state, verbose=False).fit(self.X, self.lengths)

        return best_model
