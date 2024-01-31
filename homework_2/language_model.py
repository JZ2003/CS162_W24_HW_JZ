from collections import Counter
import argparse
import math

def load_data():
    """
    Loading training and dev data. We append an eos token at the end of each sentence.
    """
    train_path = 'data/train.txt' # the data paths are hard-coded 
    dev_path  = 'data/dev.txt'
    
    with open(train_path, 'r') as f:
        train_data = [f'{l.strip()} <eos>' for l in f.readlines()]
    with open(dev_path, 'r') as f:
        dev_data = [f'{l.strip()} <eos>' for l in f.readlines()]
    return train_data, dev_data

class LanguageModel(object):
    def __init__(self, train_data, N, backoff, verbose=False):
        """
        Args:
            train_data (list of str): list of sentences comprising the training corpus.
            N (int): the order of language model to build (i.e. 1 for unigram, 2 for bigram, etc.).
            backoff (bool): whether to use backoff smoothing.
            verbose (bool): whether output information for debug.
        """
        self.N = N
        # We create a vocabulary that is a Python Counter object of (word: frequency) key-value pairs.
        self.vocab = Counter( [words for sent in train_data for words in sent.split()] )
        self.vocab_size = len(self.vocab)
        self.word_cnt = sum(self.vocab.values()) # word count can be used for unigram models
        self.backoff = backoff
        self.verbose = verbose
        if self.verbose:
            print(f'Vocabulary size: {self.vocab_size}')

        # We insert bos tokens in front of each sentence.
        # For N-gram LMs, we insert (N-1) bos token(s).
        # In this question, we do not insert any bos tokens for unigram models.
        self.bos = ' '.join(['<bos>']*(self.N-1))
        self.train_data = [f'{self.bos} {sent}' for sent in train_data]

        # We then count the frequency of N-grams.
        self.train_ngram_freq = self._get_ngrams_freq(self.train_data, N)
        
        # We create a dictionary of 1 to n-1 frequencies where the key is the n and the value is the frequency.
        self.train_ngram_backoff_freq = {}
        for n in range(1, N):
            self.train_ngram_backoff_freq[n] = self._get_ngrams_freq(self.train_data, n)

    def _get_ngrams(self, sent, n):
        """
        Given a text sentence and the argument n, we convert it to a list of n-gram tuples.

        Args:
            sent (str): input text sentence.
            n (int): the order of n-grams to return (i.e. 1 for unigram, 2 for bigram, etc.).
        Returns:
            ngrams (list of tuple): list of n-gram tuples

        Example 1:
            Input: sent="<bos> I like NLP <eos>", n=2
            Output: [("<bos>", "I"), ("I", "like"), ("like", "NLP"), ("NLP", "<eos>")]
        Example 2:
            Input: sent="<bos> I like NLP <eos>", n=1
            Output: [("<bos>",), ("I",), ("like",), ("NLP",), ("<eos>",)]
        """
        words = sent.split()
        ngrams = []
        for i in range(len(words)-n+1):
            # TODO: construct a ngram tuple with the first element being {words[i]}
            pass
        return ngrams

    def _get_ngrams_freq(self, corpus, n):
        """
        Given a training corpus, count the frequency of each n-gram.

        Args:
            corpus (list of str): list of sentences comprising the training corpus with <bos> inserted.
            n (int): the order of n-grams to count (i.e. 1 for unigram, 2 for bigram, etc.).
        Returns:
            ngram_freq (Counter): Python Counter object of (ngram (tuple), frequency (int)) key-value pairs
        """
        corpus_ngrams = []
        for sent in corpus:
            sent_ngrams = self._get_ngrams(sent, n)
            corpus_ngrams += sent_ngrams
        ngram_freq = Counter(corpus_ngrams)

        if self.verbose and n == self.N:
            print(f'Top 5 most frequent {n}-grams in the training data along with their frequency:')
            print(sorted(ngram_freq.items(), key=lambda x: x[1], reverse=True)[:5])
        return ngram_freq

    
    def sent_perplexity(self, sent):
        """
        Calculate the perplexity of the model against a given sentence
        
        Args:
            sent (str): a text sentence without <bos>.
        Returns:
            perplexity (float): the perplexity of the model as a float.
            log_prob (float): the log probability of this sentence.
            cnt_ngrams (int): the total number of n-grams in this sentence.
        """
        # We first insert bos tokens to the sentence and get its ngrams.
        sent = f'{self.bos} {sent}'
        sent_ngrams = self._get_ngrams(sent, self.N)

        log_prob = 0.0
        # We then iterate over each ngram.
        for ngram in sent_ngrams:
            if ngram in self.train_ngram_freq:
                # Case 1: this ngram appears in the training data.
                if self.N == 1:
                    # Unigram LMs are treated separately.
                    # TODO: compute the log probability.
                    pass
                else:
                    # TODO: compute the log probability for n-gram models (n>=2).
                    pass
            
            else:
                # Case 2: this ngram does not appear in the training data.
                # TODO: the instances in which the model perplexity is math.nan
                if False: # replace False
                    log_prob = math.nan
                else:
                    ngram_backoff = ngram
                    # iteratively reduce the a lower-order if that ngram does not exist
                    for n in reversed(range(1, self.N)):
                        ngram_backoff = ngram_backoff[1:]
                        if ngram_backoff in self.train_ngram_backoff_freq[n]:
                            # TODO: compute the log probaility when using backoff smooting
                            # recall there is a specical case when n=1
                            pass
                    
        cnt_ngrams = len(sent_ngrams)
        # TODO: compute sentence-level perplexity
        perplexity = None # replace None
        return perplexity, log_prob, cnt_ngrams


    def corpus_perplexity(self, corpus):
        """
        Calculate the perplexity of the model against a corpus.
        Here, we iterate over the statistics of each individual sentence,
        based on which we get the final corpus-level perplexity.
            
        Args:
            corpus (list of str): list of sentences.
        Returns:
            perplexity (float): the perplexity of the model as a float.
        """
        corpus_log_prob = 0
        corpus_cnt_ngrams = 0
        for sent in corpus:
            _, sent_log_prob, sent_cnt_ngrams = self.sent_perplexity(sent)
            corpus_log_prob += sent_log_prob
            corpus_cnt_ngrams += sent_cnt_ngrams
        # TODO: compute corpus-level perplexity. The equation should be almost the same to 
        #       sentence-level perplexity.
        perplexity = None # replace None
        return perplexity
        
    def greedy_search(self, max_steps=50):
        """
        Generate the most probable sentence by doing greedy search on the model.
            
        Args:
            max_steps (int): the maximum length of the generated sentence (default: 50).
        Returns:
            a string of the generated sentence.
        """
        # The sentence first start with the bos token(s).
        words = self.bos.split()

        while len(words) < max_steps:
            # At each step, we generate one word based on n-gram frequency.
            next_word = None
            next_max_freq = -1
            for word in self.vocab:
                # We iterate over the vocabulary and select the most probable word.
                # For each candidate word, we construct its corresponding n-gram.
                if self.N == 1:
                    ngram = tuple([word])
                else:
                    ngram = tuple(words[-self.N+1:] + [word])
                # TODO: given the n-gram, we retrieve its frequency in the training data.
                #       We update {next_word} and {next_max_freq} under certain conditions.
                pass
            if next_word is None:
                break
            words.append(next_word) 
            if next_word == '<eos>':
                break

        return ' '.join(words)
                

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, required=True,
            help='Order of N-gram model to create (i.e. 1 for unigram, 2 for bigram, etc.)')
    parser.add_argument('--b', type=bool, default=False,
            help='Parameter for backoff smoothing (default is False (disabled) -- set to True for backoff smoothing)')
    parser.add_argument('--verbose', type=bool, default=True,
            help='Will print information that is helpful for debug if set to True')
    args = parser.parse_args()

    # Loading data and display samples
    train_data, dev_data = load_data()
    if args.verbose:
        print('First 3 elements in the loaded training data:')
        print(train_data[:3])
        print('First 3 elements in the loaded dev data:')
        print(dev_data[:3])

    # Building a language model instance
    lm = LanguageModel(train_data, args.N, args.b, args.verbose)

    # Computing the perplexity of this language model on the training and dev data
    print(f'Perplexity on the training data: \n{lm.corpus_perplexity(train_data)}')
    print(f'Perplexity on the dev data: \n{lm.corpus_perplexity(dev_data)}')

    # Generating the most probable sentence using greedy search
    print(f'Generating the most probable sentence using greedy search: \n{lm.greedy_search()}')