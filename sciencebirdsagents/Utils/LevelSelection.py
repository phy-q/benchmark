class LevelSelectionSchema:
    class MaxAttempts:
        """
        return the next unsolved start_level index that the number of attempts is less than the predefined value
        """

        def __init__(self, max_attempts_value):
            self.max_attempts_value = max_attempts_value

        def select(self, total_score_record):
            unsolved_levels = list(filter(lambda x: not x[1]['did_win'], total_score_record.items()))
            filtered_levels = list(filter(lambda x: x[1]['attempts'] < self.max_attempts_value, unsolved_levels))
            if len(filtered_levels) == 0:
                return None
            idx = filtered_levels[0][0]
            return idx

    class RepeatPlay:
        '''
        return the next start_level index that the number of attempts is less than the predefined value
        '''

        def __init__(self, num_times):
            self.num_times = num_times

        def select(self, total_score_record):
            filtered_levels = list(filter(lambda x: x[1]['attempts'] < self.num_times, total_score_record.items()))
            if len(filtered_levels) == 0:
                return None
            idx = filtered_levels[0][0]
            return idx
