import pandas as pd
import numpy as np
import pickle

# Class for user profile
class Profile():
    # Similarity matrix will be based on genre and player count for now
    similarity_matrix = []
    # games_played is a dictionary where the key
    # represents the board game ID, and the value is the user's rating of that game
    games_played = {}
    games_data = None # games_data is a DataFrame containing games.csv

    def __init__(self):
        self.games_played = {}

    def fit_recommendations(self, games_played):
        # If games_played is empty, we need to throw an error
        if (games_played == {}):
            raise ValueError("games_played cannot be empty")
        self.games_played = games_played

        # Construct the similarity matrix
        self.construct_similarity_matrix()

    def update_games(self, bgg_id, score):
        self.games_played[bgg_id] = score

    def construct_similarity_matrix(self):
        # Read in games.csv
        games_df = pd.read_pickle("df_games.pickle")
        # Read in mechanics.csv
        mechanics_df = pd.read_pickle("df_mechanics.pickle")
        # New row for similarity matrix
        # Indices: Thematic, Strategy, War, Family, CGS, Abstract, Party, Childrens
        total_weight = 0
        for game in self.games_played:
            new_row = []
            row = games_df[games_df["BGGId"] == game]
            categories = ["Cat:Thematic", "Cat:Strategy", "Cat:War", "Cat:Family", 
                          "Cat:CGS", "Cat:Abstract", "Cat:Party", "Cat:Childrens"]
            game_mechanics = mechanics_df.columns[1:]
            for category in categories:
                # If the game we're on doesn't follow a certain genre, set the default value to 3
                val = row[category]
                if (int(val) == 0):
                    new_row.append(10)
                # Otherwise, set the value to the user's rating of the board game as a whole
                else:
                    new_row.append(self.games_played[game] * 5)
            # The mechanics are scaled from one to five since they're less important
            for mechanic in game_mechanics:
                mechanic_row = mechanics_df[mechanics_df["BGGId"] == game]
                if (int(mechanic_row[mechanic]) == 0):
                    new_row.append(2)
                else:
                    new_row.append(self.games_played[game])

            # Add the score weighting of the board game as just the number 20
            # TODO: make this a hyperparameter
            new_row.append(20)
            # Also make sure we have a slot for number of user ratings
            new_row.append(0.0005)
            self.similarity_matrix.append(new_row)
            total_weight += float(row["GameWeight"])
        self.avg_weight = float(total_weight / len(self.games_played))
    
    def print_similarity_matrix(self):
        for row in self.similarity_matrix:
            print(row)
    
    def recommendation_strength(self, new_board_game):
        # Convert the similarity matrix to a numpy array
        np_similarity_matrix = np.array(self.similarity_matrix)
        # Convert the board game vector to a numpy array
        np_new = np.array(new_board_game[0:len(new_board_game)-1])
        # Matrix multiplication
        result = np_similarity_matrix @ np_new
        # Return the sum of each column of the matrix as the "strength" of the recommendation
        sums = np.sum(result, axis=0)
        # Make sure to penalize games that don't align with the user's weight very much
        weights = list(new_board_game[len(new_board_game)-1])
        penalties = [abs(self.avg_weight - item) for item in weights]
        final_result = [x - (y)**7 for x, y in zip(list(sums), penalties)]
        return final_result
    
    def find_recommendations(self, n):
        # n is the number of recommendations we wish to retrieve
        games_df = pd.read_pickle("df_games.pickle")
        mechanics_df = pd.read_pickle("df_mechanics.pickle")
        # Remove games from games_df and mechanics_df that match up with what the user has played
        values_to_remove = list(self.games_played.keys())
        games_df = games_df[~games_df['BGGId'].isin(values_to_remove)]
        mechanics_df = mechanics_df[~mechanics_df['BGGId'].isin(values_to_remove)]

        input_list = [games_df["Cat:Thematic"], 
                        games_df["Cat:Strategy"], games_df["Cat:War"], 
                        games_df["Cat:Family"], games_df["Cat:CGS"], 
                        games_df["Cat:Abstract"], games_df["Cat:Party"], 
                        games_df["Cat:Childrens"]]
        # Add the game mechanics to the input_list
        input_list += [mechanics_df[i] for i in mechanics_df.columns[1:]]
        # input_list.append(games_df["Rank:reversedboardgame"])
        input_list.append(games_df["AvgRating"])
        input_list.append(games_df["NumUserRatings"])
        input_list.append(games_df["GameWeight"])
        # We also need to consider the category of board game it falls into
        games_df["Score"] = self.recommendation_strength(input_list)

        # DEBUG
        # best_games = games_df.nlargest(20, "Score")
        # for index, row in best_games.iterrows():
        #     print(row["Score"])

        top_n_recommendations = games_df.nlargest(n, "Score")
        return top_n_recommendations


def main():
    # ROOT: 237182
    # JOTL: 291457
    # Casual: 172225:10, 192291:10, 2223:10
    # Complex: 174430:10, 224517:10, 96848:10
    model = Profile()
    # Save model as a pickle
    with open("model.pkl", "wb") as file:
        pickle.dump(model, file)
    # user.fit_recommendations({174430:10, 224517:10, 96848:10})
    # user.print_similarity_matrix()
    # result = user.find_recommendations(10)
    # print(result["Name"])

if __name__ == "__main__":
    main()