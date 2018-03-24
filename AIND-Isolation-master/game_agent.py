"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    if game.is_loser(player):
        return float('-inf')
    if game.is_winner(player):
        return float('inf')
    # Calculate score based on which player moves towards center as that avoids getting trapped in corner 
    # or by a partion
    w, h = game.width, game.height
    opponent = game.get_opponent(player)

    my, mx = game.get_player_location(player)
    oy, ox = game.get_player_location(opponent)
    player_position_from_center = (my-h/2)**2 - (mx-w/2)**2
    opponent_position_from_center = (oy-h/2)**2 - (ox-w/2)**2

    return float(player_position_from_center/(1+opponent_position_from_center)**2)

def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    # Calulate score based on unique moves available to players
    # more uniquw moves means a player can make more moves without being affected by the player much
    if game.is_loser(player):
        return float('-inf')
    if game.is_winner(player):
        return float('inf')

    moves_player = game.get_legal_moves(player)
    moves_opponent = game.get_legal_moves(game.get_opponent(player))
    player_moves_unique = len([m for m in moves_player if not m in moves_opponent])*(-1)+len([m for m in moves_player if m in moves_opponent])*2
    opponent_unique_moves = len([m for m in moves_opponent if not m in moves_player])*2 - len([m for m in moves_opponent if m in moves_player])
    
    return float(player_moves_unique/(1+opponent_unique_moves**2))

def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    if game.is_loser(player):
        return float('-inf')
    if game.is_winner(player):
        return float('inf')
    # Calculate score based on open spaces as more open spaces near
    # player location means more possible moves
    open_spaces = game.get_blank_spaces()
    opponent = game.get_opponent(player)
    #player_location
    px, py = game.get_player_location(player)
    ox, oy = game.get_player_location(opponent)
    w, h = game.width, game.height
    player_open_spaces = opponent_open_spaces = 0
    
    for os_x, os_y in open_spaces:
        # open spaces for player
        if abs(os_x - px) >= 1 and abs(os_y - py) >= 1:
            if abs(os_x - 0) <= 2 or abs(os_x - w) <= 2 or abs(os_y - 0) <= 2 or abs(os_y - h) <= 2:
                # player will move near corners penalize
                player_open_spaces += 0.5
            else:
                player_open_spaces += 1
        # open spaces for opponent
        if abs(os_x - ox) >= 1  and abs(os_y - oy) >= 1:
            if abs(os_x - 0) <= 2 or abs(os_x - w) <= 2 or abs(os_y - 0) <= 2 or abs(os_y - h) <= 2:
                # opponent is near corners
                opponent_open_spaces += 0.5
            else:
                opponent_open_spaces += 1
    
    return float(player_open_spaces/(1+opponent_open_spaces))


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # TODO: finish this function!
        def min_value(game, depth):
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()
            moves = game.get_legal_moves()
            if depth <= 0 or len(moves) == 0 :
                return self.score(game, self)
            score = float('inf')
            for m in moves:
                score = min(score, max_value(game.forecast_move(m), depth - 1))
            return score
        def max_value(game, depth):
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()
            moves = game.get_legal_moves()
            if depth <= 0 or len(moves) == 0 :
                return self.score(game, self)
            score = float('-inf')
            for m in moves:
                score = max(score, min_value(game.forecast_move(m), depth - 1))
            return score

        moves = game.get_legal_moves()
        if depth <= 0 or len(moves) == 0 :
                return (-1,-1)
        best_move = moves[0]
        best_score = float('-inf')
        for m in moves:
            score = min_value(game.forecast_move(m), depth - 1)
            if score > best_score:
                best_score = score
                best_move = m
        return best_move

class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # TODO: finish this function!
        best_move = (-1,-1)
        depth = 1
        try:
            while True:
                best_move = self.alphabeta(game, depth)
                #print(self.search_depth, best_move)                
                depth += 1 
        except SearchTimeout:
            #print('SearchTimeout')
            pass
        #print('Best Move ',best_move)
        return best_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # TODO: finish this function!
        def min_value(game, depth, alpha, beta):
            if self.time_left() < self.TIMER_THRESHOLD:
                #print('alphabeta min timeout', self.time_left(), self.TIMER_THRESHOLD)
                raise SearchTimeout()
            moves = game.get_legal_moves()
            if depth <= 0 or len(moves) == 0 :
                return self.score(game,self)            
            best_score = float('inf')
            for move in moves:
                # get max score of the moves
                score = max_value(game.forecast_move(move), depth - 1,alpha,beta)
                # update best score if new score is less
                best_score = min(score,best_score)
                if best_score <= alpha:
                    return best_score
                beta = min(best_score, beta)
            return best_score            
        
        def max_value(game, depth, alpha, beta):
            if self.time_left() < self.TIMER_THRESHOLD:
                #print('alphabeta max timeout')
                raise SearchTimeout()
            moves = game.get_legal_moves()
            if depth <= 0 or len(moves) == 0 :
                return self.score(game, self)
            best_score = float('-inf')
            for move in moves:
                score = min_value(game.forecast_move(move), depth - 1, alpha, beta)
                best_score = max(best_score, score)
                if best_score >= beta:
                    return best_score
                alpha = max(best_score,alpha)
            return best_score

        moves = game.get_legal_moves()
        if len(moves) == 0:
            return (-1,-1)
        best_score = float('-inf')
        best_move = moves[0]
        for move in moves:
            score = min_value(game.forecast_move(move), depth -1, alpha, beta)
            if best_score < score:
                best_score = score
                best_move = move
                alpha = max(alpha,best_score)
        return best_move