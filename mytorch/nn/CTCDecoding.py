import numpy as np

class GreedySearchDecoder(object):

    def __init__(self, symbol_set):
        """
        
        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        """

        self.symbol_set = symbol_set


    def decode(self, y_probs):
        """

        Perform greedy search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
            batch size for part 1 will remain 1, but if you plan to use your
            implementation for part 2 you need to incorporate batch_size

        Returns
        -------

        decoded_path [str]:
            compressed symbol sequence i.e. without blanks or repeated symbols

        path_prob [float]:
            forward probability of the greedy path

        """

       

        # TODO:
        # 1. Iterate over sequence length - len(y_probs[0])
        # 2. Iterate over symbol probabilities
        # 3. update path probability, by multiplying with the current max probability
        # 4. Select most probable symbol and append to decoded_path
        # 5. Compress sequence (Inside or outside the loop)

        #return decoded_path, path_prob
        def str_path(path):
            path = str(path).replace("'","").replace(",","").replace(" ","").replace("[","").replace("]","")
            
            return path

        decoded_path = []
        blank = 0
        path_prob = 1

        for t in range(y_probs.shape[1]):
            max_prob_idx = np.argmax(y_probs[:, t, 0])
            path_prob *= y_probs[max_prob_idx, t, 0]
            if max_prob_idx != blank and (not decoded_path or max_prob_idx != decoded_path[-1]):
                decoded_path.append(max_prob_idx)

        decoded_path = ''.join([self.symbol_set[sym - 1] for sym in decoded_path]) if decoded_path else ''

        return decoded_path, path_prob

        


class BeamSearchDecoder(object):

    def __init__(self, symbol_set, beam_width):
        """

        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        beam_width [int]:
            beam width for selecting top-k hypotheses for expansion

        """

        self.symbol_set = symbol_set
        self.beam_width = beam_width

    def decode(self, y_probs):

        """
        
        Perform beam search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
            batch size for part 1 will remain 1, but if you plan to use your
            implementation for part 2 you need to incorporate batch_size

        Returns
        -------
        
        forward_path [str]:
            the symbol sequence with the best path score (forward probability)

        merged_path_scores [dict]:
            all the final merged paths with their scores
        """    

        

        def str_path(path):
            return ''.join(path.split())

        T = y_probs.shape[1]
        best_path, final_path_scores = None, None
        
        y_probs = y_probs[:, :, 0]
        global path_score
        global blank_path_score
        path_score = []
        blank_path_score = []
        new_paths_with_terminal_blank, new_paths_with_terminal_symbol, new_blank_path_score, new_path_score = self._initialize_paths(self.symbol_set, y_probs[:, 0])

        for t in range(1, len(y_probs[1])):
            paths_with_terminal_blank, paths_with_terminal_symbol, blank_path_score, path_score = self._prune(new_paths_with_terminal_blank, new_paths_with_terminal_symbol, new_blank_path_score, new_path_score, 10000 if t == 1 else self.beam_width)
            new_paths_with_terminal_blank, new_blank_path_score = self._extend_with_blank(paths_with_terminal_blank, paths_with_terminal_symbol, y_probs[:, t])
            new_paths_with_terminal_symbol, new_path_score = self._extend_with_symbol(paths_with_terminal_blank, paths_with_terminal_symbol, self.symbol_set, y_probs[:, t])

        merged_paths, final_path_score = self._merge_identical_paths(new_paths_with_terminal_blank, new_blank_path_score, new_paths_with_terminal_symbol, new_path_score)
        best_score = 0

        for path, score in final_path_score.items():
            if score > best_score:
                best_score = score
                best_path = path

        return best_path, final_path_score

    def _initialize_paths(self, symbol_set, y):
        initial_blank_path_score = {}
        initial_path_score = {}

        path = "-"
        initial_blank_path_score[path] = y[0]
        initial_paths_with_final_blank = [path]

        initial_paths_with_final_symbol = []

        for index, c in enumerate(symbol_set):
            path = c
            initial_path_score[path] = y[index + 1]
            initial_paths_with_final_symbol.append(path)

        return initial_paths_with_final_blank, initial_paths_with_final_symbol, initial_blank_path_score, initial_path_score

    def _extend_with_blank(self, paths_with_terminal_blank, paths_with_terminal_symbol, y_pred):
        updated_paths_with_terminal_blank = []
        updated_blank_path_score = dict()

        for path in paths_with_terminal_blank:
            updated_paths_with_terminal_blank.append(path)
            updated_blank_path_score[path] = blank_path_score[path] * y_pred[0]

        for path in paths_with_terminal_symbol:
            if path in updated_paths_with_terminal_blank:
                updated_blank_path_score[path] += path_score[path] * y_pred[0]
            else:
                updated_paths_with_terminal_blank.append(path)
                updated_blank_path_score[path] = path_score[path] * y_pred[0]
        return updated_paths_with_terminal_blank, updated_blank_path_score
    
    def _extend_with_symbol(self, paths_with_terminal_blank, paths_with_terminal_symbol, symbol_set, y):
        updated_paths_with_terminal_symbol = []
        updated_path_score = dict()

        for path in paths_with_terminal_blank:
            for i, symb in enumerate(symbol_set):
                new_path = path + symb
                if path == "-":
                    new_path = symb
                updated_paths_with_terminal_symbol.append(new_path)
                updated_path_score[new_path] = blank_path_score[path] * y[i + 1]

        for path in paths_with_terminal_symbol:
            for index, c in enumerate(symbol_set):
                new_path= path if c == path[-1] else path + c
            if new_path in updated_paths_with_terminal_symbol:
                updated_path_score[new_path] += path_score[path] * y[index + 1]
            else:
                updated_paths_with_terminal_symbol.append(new_path)
                updated_path_score[new_path] = path_score[path] * y[index + 1]
        return updated_paths_with_terminal_symbol, updated_path_score
    

    def _prune(self, paths_with_terminal_blank, paths_with_terminal_symbol, blank_path_score, path_score, beam_width):
        pruned_blank_path_score = dict()
        pruned_path_score = dict()
        scores = []

        for i in paths_with_terminal_blank:
            scores.append(blank_path_score[i])
        for i in paths_with_terminal_symbol:
            scores.append(path_score[i])

        scores = sorted(scores, reverse=True)

        cutoff = []

        if beam_width < len(scores):
            cutoff = scores[beam_width - 1]
        else:
            cutoff = scores[-1]

        pruned_paths_with_terminal_blank = []

        for i in paths_with_terminal_blank:
            if blank_path_score[i] >= cutoff:
                pruned_paths_with_terminal_blank.append(i)

        for p in paths_with_terminal_blank:
            if blank_path_score[p] >= cutoff:
                pruned_blank_path_score[p] = blank_path_score[p]
        pruned_paths_with_terminal_symbol = []

        [pruned_paths_with_terminal_symbol.append(i) for i in paths_with_terminal_symbol if path_score[i] >= cutoff]
        for i in paths_with_terminal_symbol:
            if path_score[i] >= cutoff:
                pruned_path_score[i] = path_score[i]

        return pruned_paths_with_terminal_blank, pruned_paths_with_terminal_symbol, pruned_blank_path_score, pruned_path_score

    def _merge_identical_paths(self, paths_with_terminal_blank, blank_path_score, paths_with_terminal_symbol, path_score):
        merged_paths = paths_with_terminal_symbol
        final_path_score = path_score

        for path in paths_with_terminal_blank:
            if path not in merged_paths:
                merged_paths.append(path)
            final_path_score[path] = blank_path_score[path]
        else:
            final_path_score[path] += blank_path_score[path]

        return merged_paths, final_path_score

