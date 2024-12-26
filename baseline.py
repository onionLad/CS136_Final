#
#   baseline.py
#   Bill Xia
#   CS 136: Statistical Pattern Recognition
#   Created: 4/23/24
#
# Purpose: Implements a class that contains all the functionalities of the
#          baseline model of our biomedical term identifier. 
#

# Globals
NUM_LABELS = 3

# Imports
import numpy as np

# Class
class IdentifierBaseline:
    '''
    A baseline model for the biomedical term identifier. This model predicts
    term labels using a Categorical PMF optimized over maximum likelihood.
    Label predictions are based on the previous label and the current word.
    '''

    # Constructor ----------------------------------------------------------- #
    def __init__(self):
        '''
        Constructor for IdentifierBaseline.
        '''

        # A list of words that our identifier has seen before. We initialize
        # with '' as the first element to represent unseen words.
        self.vocabulary = ['']

        # to_int converters. These functions convert words/labels to integer
        # representations used as indices in our count and frequency tables.
        self.label_to_int = lambda l: {'B': 1, 'I': 2, 'O': 0}[l]
        self.word_to_int  = lambda w: self.vocabulary.index(w.lower().translate(str.maketrans('', '', ",.:")))
    # ----------------------------------------------------------------------- #

    # Training -------------------------------------------------------------- #
    def __build_vocabulary(self, training_data: list[tuple[str, list[str]]]):
        '''
        Private method that builds the baseline's vocabulary.
        '''
        for sent, _ in training_data:

            # Make sent lowercase, remove extra spaces, split into words.
            sent_split = sent.lower().strip().split()

            for w in sent_split:

                # Remove punctuation and add to vocabulary if it's not already
                # there.
                w_cleaned = w.translate(str.maketrans('', '', ",.:"))
                if w_cleaned not in self.vocabulary:
                    self.vocabulary.append(w_cleaned)

    def __build_prob_tables(self, training_data: list[tuple[str, list[str]]]):
        '''
        Private method that builds the baseline's probability tables.
        '''

        # Start by building the counts tables. We initialize counts to 1s for
        # pseudocounts.
        zeta_counts = np.ones((len(self.vocabulary), NUM_LABELS))
        eta_counts  = np.ones((NUM_LABELS, len(self.vocabulary), NUM_LABELS))
        for sent, labels in training_data:

            # Pre-process sent and zip with word labels.
            sent_split = sent.lower().strip().split()
            sl_zipped  = list(zip(sent_split, labels))

            # Iterate all but the final pair in the zipped list.
            for idx, (w, l) in enumerate(sl_zipped[:len(sl_zipped)-1]):

                # First word in sent is counted for the zeta table.
                if idx == 0:
                    zeta_counts[self.word_to_int(w),
                                self.label_to_int(l)] += 1

                # Get next word and label.
                next_w, next_l = sl_zipped[idx+1]

                # Store counts according to current label and next word-label
                # pair.
                eta_counts[self.label_to_int(l),
                           self.word_to_int(next_w),
                           self.label_to_int(next_l)] += 1

        # Next building the zeta frequency table.
        self.zeta = np.empty_like(zeta_counts)
        for idx in range(len(self.vocabulary)):

            # We get frequencies by dividing counts by total counts for that
            # context window.
            curr_counts = zeta_counts[idx]
            count_total = np.sum(curr_counts)
            for jdx in range(NUM_LABELS):
                self.zeta[idx, jdx] = curr_counts[jdx] / count_total

        # Finally build the eta frequency table.
        self.eta = np.empty_like(eta_counts)
        for idx in range(NUM_LABELS):
            for jdx in range(len(self.vocabulary)):

                # Get frequencies.
                curr_counts = eta_counts[idx, jdx]
                count_total = np.sum(curr_counts)
                for kdx in range(NUM_LABELS):
                    self.eta[idx, jdx, kdx] = curr_counts[kdx] / count_total

        # print(self.eta[:,0])

    def train(self, training_data: list[tuple[str, list[str]]]) -> None:
        '''
        Training method for the IdentifierBaseline class. This function builds
        a table of probability vectors that can be used to predict word labels.

        #### Arguments
        `training_data`: a list of sentences (string) paired with word labels
        (list of strings).
        '''

        # Start by building the identifier's vocabulary.
        self.__build_vocabulary(training_data)

        # Build zeta and eta.
        self.__build_prob_tables(training_data)
    # ----------------------------------------------------------------------- #

    # Predicting ------------------------------------------------------------ #
    def __get_zeta(self, w: str):
        '''
        Private method that gets the appropriate zeta vector for a given word.
        '''
        if w in self.vocabulary:
            return self.zeta[self.word_to_int(w)]
        else:
            return self.zeta[self.word_to_int('')]

    def __get_eta(self, pl: int, w: str):
        '''
        Private method that gets the appropriate eta vector for a given prev
        label and curr word.
        '''
        if w in self.vocabulary:
            return self.eta[pl, self.word_to_int(w)]
        else:
            return self.eta[pl, self.word_to_int('')]

    def __CatPMF(self, l, mu):
        '''
        Private method that computes the likelihood that x appears given
        probability vector mu in a Categorical model. 
        '''
        prod = 1
        for idx in range(len(mu)):
            if l == idx:
                prod *= mu[idx]
        return prod

    def predict(self, sent: str) -> list[int]:
        '''
        Prediction method for the IdentifierBaseline class. This function
        returns a list of integer representations of label predictions for
        each word in the sentence.
        '''

        # Intialize lists of words and label predictions.
        sent_split  = sent.lower().strip().split()
        label_preds = []

        # Word statistics.
        unseen_words = 0.0
        total_words  = 0.0

        for idx, w in enumerate(sent_split):

            if w.translate(str.maketrans('', '', ",.:")) not in self.vocabulary:
                unseen_words += 1
            total_words += 1

            # Process word differently if it's the first one in the sentence.
            if idx == 0:

                curr_zeta = self.__get_zeta(w)
                obj_func = lambda l: self.__CatPMF(l, curr_zeta)
                best_label = np.argmax( [obj_func(l) for l in [0, 1, 2]] )
                label_preds.append(best_label)

                # print(curr_zeta)

                continue

            # Get previous word and label.
            prev_l = label_preds[idx-1]

            # get curr eta and compute best label.
            curr_eta = self.__get_eta(prev_l, w)
            obj_func = lambda l: self.__CatPMF(l, curr_eta)
            best_label = np.argmax( [obj_func(l) for l in [0, 1, 2]] )
            label_preds.append(best_label)

            # print(curr_eta)

        return label_preds, unseen_words, total_words
    # ----------------------------------------------------------------------- #



