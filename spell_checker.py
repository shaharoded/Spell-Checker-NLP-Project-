'''
TO-DO (Further Improvements):
 - Incorporate Wikipedia errors into error table
 - remmember to add #c formatted errors to error table - and look for potencial errors in pairs identification.
 - Train the model on a bigger, more diverse corpus.
'''

import re
import random
import math
from collections import defaultdict, Counter
import nltk

import string


class Spell_Checker:
    """
    The class implements a context sensitive spell checker. The corrections
    are done in the Noisy Channel framework, based on a language model and
    an error distribution model.
    """

    def __init__(self,  lm=None):
        """
        Initializing a spell checker object with a language model (LM) as an
        instance  variable.

        Args:
            lm: a language model object. Defaults to None.
        """
        self.lm = lm
        self.error_tables = None
                

    def add_language_model(self, lm):
        """
        Adds the specified language model as an instance variable.
        (Replaces an older LM dictionary if set).
        Updates the error probabilities based on the new LM.
        
        NOTE: 
        LM must be of class Spell_Checker.Language_Model to assure all mandatory
        attributes were created during the initialization process.

        Args:
            lm: a Spell_Checker.Language_Model object
        """
        self.lm = lm
        

    def add_error_tables(self, error_tables):
        """ 
        Adds the specified dictionary of error tables as an instance variable.
        (Replaces an older value dictionary if set).

        Args:
        error_tables (dict) - frequency: a dictionary of error tables in the format
        of the provided confusion matrices:
        https://www.dropbox.com/s/ic40soda29emt4a/spelling_confusion_matrices.py?dl=0
        Saved as spelling_confusion_matrices.py in this repository.
        """
        
        self.error_tables = error_tables


    def evaluate_text(self, text):
        """
        Returns the log-likelihood of the specified text given the language
        model in use. Smoothing should be applied on texts containing OOV words
        (handeled by the language model).
        The function is used to assess the possibility of a sentence to be derived from 
        the given LM.          

        Args:
            text (str): Text to evaluate.

        Returns:
            Float (negative): The float should reflect the (log) probability, 
            based on the LM.
        """
        if not self.lm:
            raise ValueError("Language model is not set.")
        return self.lm.evaluate_text(text)


    def spell_check(self, text, alpha, k=1):
        """ 
        Returns the most probable fix for the specified text. Use a simple
        noisy channel model if the number of tokens in the specified text is
        smaller than the length (n) of the language model.

        Args:
            text (str): The text to spell check.
            alpha (float): The probability of keeping a lexical word as is.
                            In practice, thats the probability to keep it as is, if it exists 
                            in the vocabulary.
            k (int): Max number of words to correct. Will be handeled with a loop, 
                    each iteration will suggest the most probable sentence. If the corrected
                    sentence == original sentence - abort loop. 
                    Initiated to 1 - fix the most significant error.

        Return:
            A normalized modified string (or a copy of the original if no corrections are made.)
        """
        # Normalize the text
        sentence = normalize_text(text)
        most_probable_sentence = None
        
        for _ in range(k):
            # Tokenize the text
            words = sentence.split()
        
            candidate_sentences = []

            for i, word in enumerate(words):
                # Get the context for the word (up to the n-gram window size)
                context_start_index = max(0, i - self.lm.get_model_window_size() + 1)
                context = ' '.join(words[context_start_index:i])

                # Generate correction for the word considering the context
                # correction can be the same word
                corrected_word = self._correction(word, context, alpha)
                # Add only modifications, where the word can be found in te model
                if corrected_word != word and corrected_word in self.lm.vocabulary:                    
                    words[i] = corrected_word
                    candidate_sentence = ' '.join(words)
                    candidate_sentences.append(candidate_sentence)
                
            # Evaluate each candidate sentence and pick the best one
            if len(candidate_sentences) > 0:
                scored_sentences = [(self.evaluate_text(sent), sent) for sent in candidate_sentences]
                most_probable_sentence = max(scored_sentences, key=lambda x: x[0])[1]

                # If the most probable sentence is the same as the current sentence, break the loop
                if most_probable_sentence == sentence:
                    break

                # Update the sentence for the next iteration
                sentence = most_probable_sentence
            
        return most_probable_sentence if most_probable_sentence else sentence
    

    def _correction(self, word, context, alpha):
        """
        Generates all candidate corrections for a word.
        
        Args:
        word (str) - The original word
        context (str) - The context in which this word is passed in the sentence.
        alpha (float): The probability of keeping a lexical word as is.
        
        Return: The most probable spelling correction (a word) for 'word' 
        out of all candidates.
        """
        candidates = self._candidates(word)
        return max(candidates, key=lambda w: self._P(w, word, context, alpha))
    

    def _candidates(self, word):
        """
        Generate possible spelling corrections for word.
        Checks if the word is knows in the LM
        Generate Possible corrections that are with edit distance 1
        Generate Possible corrections that are with edit distance 2
        
        Args:
            word (str): A word for repacement.
        
        Return: An array of possible candidates for corrections + the original word.
        """
        return (self._known([word]) or
                self._known(self._edits1(word)) or
                self._known(self._edits2(word)) or
                [word])
        

    def _known(self, words):
        """
        The subset of `words` that appear in the vocabulary of the LM.
        
        Args:
            words (array): An array of soon-to-be-candidate words.
        
        Return: A subset of said array.
        """
        if not self.lm:
            return set()
        return set(w for w in words if w in self.lm.vocabulary)
    

    def _edits1(self, word):
        """
        All edits (not necassary in language) that are one edit away from `word`.
        
        Args:
            word (str): A word for repacement.
            
        Return: set()
        """
        letters    = 'abcdefghijklmnopqrstuvwxyz'
        splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
        deletes    = [L + R[1:]               for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
        replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
        inserts    = [L + c + R               for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)
    

    def _edits2(self, word):
        """
        All edits (not necassary in language) that are 2 edits away from `word`.
        
        Args:
            word (str): A word for repacement.
            
        Return: set()
        """
        return (e2 for e1 in self._edits1(word) for e2 in self._edits1(e1))
    

    def _P(self, candidate, word, context, alpha):
        """
        Computes the probability of a candidate correction given the original 'word'.
            - Calculates the probability that candidate is in the language
            - Calculates the error probability from 'word' -> 'candidate'
            - Combine the probabilities using 'alpha'
        Probability is multiplied by const 1e8 to avoid floating point errors.
        
        Args:
            candidate (str) - A candidate word to replace 'word'
            word (str) - The original word
            context (str) - The context in which the word appears (n-1 prior words)
            alpha (float) - The probability of keeping a lexical word as is. Passed as
                            error_probability if candidate == word.
        
        Return: Float, probability.
        """
        log_likelihood = self.evaluate_text(context + ' ' + candidate)  # Language model probability
        lm_prob = math.exp(log_likelihood)  # Back to probability
        error_prob = self._error_probability(candidate, word, alpha)  # Error model probability
        return 1e8*(lm_prob * error_prob) 
    
    
    def _error_probability(self, candidate, word, alpha):
        """
        Calculate the probability of transforming `word` to `candidate`.
        If word == candidate, probably the word should not be changed 
        (was offered and exists in the dictionary), thus keeping the original word.
        Will differenciate the different types of errors before calculation.
        
        Function will take words of edit distance 2 from _known(candidate) by looking 
        at all of their distance 1 edits, and than look for the max probability for them.
        
        Error probability should never be 0.
        
        Args:
            candidate (str) - A candidate word to replace 'word'
            word (str) - The original word
            alpha (float) - The probability of keeping a lexical word as is. Passed as
                            error_probability if candidate == word.
        
        Return: Float, probability.
        """
        if not self.error_tables or word == candidate:
            return alpha  # If the word is the same, no error occurred.

        # Initialize the probabilities
        insertion_prob = 0.0
        deletion_prob = 0.0
        substitution_prob = 0.0
        transposition_prob = 0.0

        # Identify the type of edit and calculate the corresponding probability
        if len(candidate) + 1 == len(word):  # Possible insertion
            insertion_prob = self._insertion_probability(candidate, word)
        elif len(candidate) == len(word) + 1:  # Possible deletion
            deletion_prob = self._deletion_probability(candidate, word)
        elif len(candidate) == len(word):  # Possible substitution or transposition
            if any(candidate[i] != word[i] for i in range(len(word))):
                substitution_prob = self._substitution_probability(candidate, word)
            if any(candidate[i] == word[i+1] and candidate[i+1] == word[i] for i in range(len(word)-1)):
                transposition_prob = self._transposition_probability(candidate, word)

        for intermediate in self._edits1(word):
            if intermediate == candidate:
                continue
            elif len(candidate) + 1 == len(intermediate):
                deletion_prob = max(deletion_prob, self._deletion_probability(candidate, intermediate))
            elif len(candidate) == len(intermediate) + 1:
                insertion_prob = max(insertion_prob, self._insertion_probability(candidate, intermediate))
            elif len(candidate) == len(intermediate):
                substitution_prob = max(substitution_prob, self._substitution_probability(candidate, intermediate))
                transposition_prob = max(transposition_prob, self._transposition_probability(candidate, intermediate))

        # Return the highest probability among the calculated probabilities
        return max(insertion_prob, deletion_prob, substitution_prob, transposition_prob, 1e-8)

    
    def _insertion_probability(self, candidate, word):
        """
        Calculate the probability of an insertion error:
        candidate = word[i] + word[i+1] for some i
        
        Args:
            candidate (str) - A candidate word to replace 'word'
            word (str) - The original word
        
        Return: Float, probability.
        """
        for i in range(len(word)):
            if word[:i] + word[i+1:] == candidate:
                pair = word[i - 1 : i+1] if i > 0 else '#' + word[i]
                pair_proba = self.error_tables['insertion'].get(pair, 0.0)
                denominator = self.lm.singulars.get(pair[0], 0.0)  # based on noisy-channel
                if denominator > 0:
                    return pair_proba/denominator
                else:
                    return 0.0
        return 0.0


    def _deletion_probability(self, candidate, word):
        """
        Calculate the probability of a deletion error.
        candidate = word[i] + c + word[i+1] for some i, c
        
        Args:
            candidate (str) - A candidate word to replace 'word'
            word (str) - The original word
        
        Return: Float, probability.
        """
        for i in range(len(candidate)):
            if candidate[:i] + candidate[i+1:] == word:
                pair = candidate[i-1:i+1] if i>0 else '#' + candidate[i]
                pair_proba = self.error_tables['deletion'].get(pair, 0.0)
                denominator = self.lm.pairs.get(pair, 0.0)  # based on noisy-channel
                if denominator > 0:
                    return pair_proba/denominator
                else:
                    return 0.0
        return 0.0


    def _substitution_probability(self, candidate, word):
        """
        Calculate the probability of a substitution error.
        candidate = word where word[i] = c for some i, c
        
        Args:
            candidate (str) - A candidate word to replace 'word'
            word (str) - The original word
        
        Return: Float, probability.
        """
        for i in range(len(word)):
            if candidate[i] != word[i]:
                pair = word[i] + candidate[i]
                pair_proba = self.error_tables['substitution'].get(pair, 0.0)
                denominator = self.lm.singulars.get(pair[1], 0.0)  # based on noisy-channel
                if denominator > 0:
                    return pair_proba/denominator
                else:
                    return 0.0
        return 0.0


    def _transposition_probability(self, candidate, word):
        """
        Calculate the probability of a transposition error.
        candidate = word[:i] + word[i + 1] + word[i] + word[i+2:] for some i
        ABDC -> ABCD
        
        Args:
            candidate (str) - A candidate word to replace 'word'
            word (str) - The original word
        
        Return: Float, probability.
        """
        for i in range(len(word) - 1):
            if (word[i] == candidate[i+1] and word[i+1] == candidate[i]
                and word[i] != candidate[i]
                    and word[:i] + word[i+2:] == candidate[:i] + candidate[i+2:]):
                pair = word[i:i+2]
                pair_proba = self.error_tables['transposition'].get(pair, 0.0)
                denominator = self.lm.pairs.get(pair, 0.0)  # based on noisy-channel
                if denominator > 0:
                    return pair_proba/denominator
                else:
                    return 0.0
        return 0.0


    #####################################################################
    #                   Inner class                                     #
    #####################################################################

    class Language_Model:
        """
        The class implements a Markov Language Model that learns a model from a given text.
        It supports language generation and the evaluation of a given string.
        The class can be applied on both word level and character level.
        """

        def __init__(self, n=3, chars=False):
            """
            Initializing a language model object.
            Args:
                n (int): the length of the markov unit (the n of the n-gram). Defaults to 3.
                chars (bool): True iff the model consists of ngrams of character level rather than word level tokens.
                                Defaults to False
            """
            self.n = n
            self.chars = chars
            self.vocabulary = set()
            self.pairs = Counter() #a dictionary of the form {char_pair (str):count}
            self.singulars = Counter() #a dictionary of the form {char (str):count}
            self.model_dict = defaultdict(Counter) #a dictionary of the form {ngram (tup):count}, holding counts of all ngrams in the specified text.
            
            # NOTE: This dictionary format is inefficient and insufficient (why?), therefore  you can (even encouraged to)
            # use a better data structure.
            # However, you are requested to support this format for two reasons:
            # (1) It is very straight forward and force you to understand the logic behind LM, and
            # (2) It serves as the normal form for the LM so we can call get_model_dictionary() and peek into you model.


        def build_model(self, text):
            """
            Populates the instance variable model_dict.
            Applies context to each token, using self.n-grams.
            Splits for tokens by character level or word level, depends on initialization.

            Will update vocabulary, pairs and singulars attributes in the model.
            pairs - Counter() of letter pairs in the text.
            singulars - Counter() of single letters appearance in the text.
            vocabulary - Set(), all unique words in the text.
            
            Args:
                text (str): the text to construct the model from.
            """
            text = normalize_text(text)  # Normalize text before processing
                
            # Count single pairs of letters frequency (for errors probability)
            for i in range(len(text) - 1):
                self.singulars[text[i]] += 1
                pair = text[i:i + 2]
                self.pairs[pair] += 1
            self.singulars[text[-1]] += 1
                
            tokens = text if self.chars else text.split()
            # Add unique tokens to the model
            self.vocabulary.update(tokens)
            
            # Update model_dict (ngrams)
            for i in range(len(tokens) - self.n):
                context = tuple(tokens[i:i+self.n])
                next_token = tokens[i+self.n]
                self.model_dict[context][next_token] += 1
            

        def get_model_dictionary(self):
            """
            Return: The dictionary class object
            """
            return self.model_dict

        def get_model_window_size(self):
            """
            Return: int, The size of the context window (the n in "n-gram")
            """
            return self.n

        def generate(self, context=None, n=20):
            """
            Returns a string of the specified length, generated by applying the language model
            to the specified seed context. If no context is specified the context should be sampled
            from the models' contexts distribution. Generation should stop before the n'th word if the
            contexts are exhausted. If the length of the specified context exceeds (or equal to)
            the specified n, the method should return a prefix of length n of the specified context.

            Args:
                context (str): a seed context to start the generated string from. Defaults to None
                n (int): the length of the string to be generated.

            Return:
                String. The generated text.
            """
            model = self.model_dict
            original_context = context  # Keep the original context

            if context is not None:
                # Normalize the context
                norm_context = normalize_text(context)
                if not self.chars:
                    norm_context = tuple(context.split()[-self.n:])
                else:
                    norm_context = tuple(norm_context[-self.n:])
            else:
                norm_context = random.choice(list(model.keys()))  # Random seed

            output = list(norm_context)
            for _ in range(n - len(norm_context)):
                if norm_context in model:
                    next_token = random.choices(list(model[norm_context].keys()), weights=model[norm_context].values())[0]
                    output.append(next_token)
                    norm_context = tuple(output[-self.n:])
                else:
                    # Context is exhausted or not found
                    break
            
            generated_text = ''.join(output) if self.chars else ' '.join(output)
            return (original_context + ' ' + generated_text[len(original_context):]).strip() if context else generated_text
            

        def evaluate_text(self, text):
            """
            Returns the log-likelihood of the specified text to be a product of the model.
            Laplace smoothing should be applied if necessary.
            Text is normalized before evaluation.
            It's the combination of the probabilities of all ngrams in the text

            Args:
                text (str): Text to evaluate.

            Returns:
                Float (negative). The float should reflect the (log) probability.
            """
            text = normalize_text(text)
            tokens = list(text) if self.chars else text.split()
            if len(tokens) < self.n:
                # If text is shorter than the n-gram size, return log probability of the whole text
                ngram = ' '.join(tokens)
                probability = self.smooth(ngram)
                return math.log(probability)
            
            log_likelihood = 0
            for i in range(len(tokens) - self.n):
                ngram = ' '.join(tokens[i:i+self.n+1])
                probability = self.smooth(ngram)
                log_likelihood += math.log(probability)
            return log_likelihood

        def smooth(self, ngram, k=1):
            """
            Returns the smoothed (Laplace) probability of the specified ngram.
            Should not be needed for generating text, but will be useful for the correction
            based on self model.
            For an attemp to get probability for len(text) < window_size, p = 1e-8

            Args:
                ngram (str): The ngram to have its probability smoothed.
                            Will be turned to tuple for this function's usage
                k (int): The factor for the laplace smoothing.

            Returns:
                float. The smoothed probability.
            """
            ngram = tuple(ngram.split())  
            context = ngram[:-1]
            token = ngram[-1]
            token_count = self.model_dict[context].get(token, 0)
            total_context_count = sum(self.model_dict[context].values())
            vocabulary_size = len(self.vocabulary)
            # Ensure a minimum probability to prevent log(0)
            smoothed_probability = (token_count + k) / (total_context_count + k * vocabulary_size)
            return max(smoothed_probability, 1e-8)  # Prevents math domain error in log



def normalize_text(text, lower = True, remove_numbers=True, remove_punctuation=False, remove_stopwords=False):
    """Returns a normalized version of the specified string. Will lower all letters, remove any HTML tags, 
    replace potentially bad characters and normalize whitespaces.

    Args:
        text (str): the text to normalize
        lower (bool): Whether to lower the rext. Defaults to True.
        remove_numbers (bool): Whether to remove numbers \ dates from the text. Defaults to True.
        remove_punctuation (bool): Whether to remove punctuation from the text. Defaults to True.
        remove_stopwords (bool): Whether to remove stopwords from the original text, and keep only meaningful words.
                                Defaults to False, for context and negations. By choosing this option you'll also remove 
                                punctuation from the sentence.
    Returns:
        str: The normalized text.
    """
    # Replacement mappings
    replacements = {
        '“': '"', '”': '"', '‘': "'", '’': "'",
        'â\x80\x9c': '"', 'â\x80\x9d': '"', 'â\x80\x99': "'"
    }
    for bad_char, good_char in replacements.items():
        text = text.replace(bad_char, good_char)

    # Remove HTML tags
    text = re.sub('<[^<]+?>', '', text)
    
    # Convert to lowercase
    if lower:
        text = text.lower()

    # Handling numbers and dates to 1 form
    # Regex to replace standalone numbers (including decimals) and percentages
    if remove_numbers:
        text = re.sub(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', 'DATE', text)  # Dates like 01/01/2020 or 2020-01-01
        text = re.sub(r'\b\d{2,4}[/-]\d{1,2}[/-]\d{1,2}\b', 'DATE', text)  # Dates like 01/01/2020 or 2020-01-01
        text = re.sub(r'\b\d+\.\d+%?\b', 'DECIMAL', text)  # Decimal numbers, optionally followed by a percent sign
        text = re.sub(r'\b\d+%?\b', 'NUM', text)  # Whole numbers, optionally followed by a percent sign
    
    if remove_punctuation:
        words = [word.strip(string.punctuation) for word in text.split()]
        text = ' '.join(words)
    
    # Remove stopwords & Normalize whitespace
    if remove_stopwords:
        # Package only needed here
        from nltk.corpus import stopwords
        nltk.download('stopwords')
        
        text = ' '.join([word.strip(string.punctuation) for word in text.split() if word not in stopwords.words('english')]) 
    else:
        # Normalize whitespace
        text = ' '.join([word.strip(string.punctuation) for word in text.split()]) 

    return text


def who_am_i():  # this is not a class method
    """
    Returns a dictionary with your name, id number and email. keys=['name', 'id','email'].
    """
    return {'name': 'Shahar Oded', 'id': '208388918', 'email': 'odedshah@post.bgu.ac.il'}