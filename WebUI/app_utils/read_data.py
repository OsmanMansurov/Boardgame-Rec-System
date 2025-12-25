import pandas as pd
import pickle

def main():
    df_games = pd.read_csv("./raw_data/games.csv")
    # Drop columns with too many null or zero values
    df_games.drop("ComAgeRec", axis=1, inplace=True)
    df_games.drop("LanguageEase", axis=1, inplace=True)
    df_games.drop("Family", axis=1, inplace=True)
    df_games.drop("NumComments", axis=1, inplace=True)

    # Drop any N/A values from df_games
    df_games = df_games.dropna()

    # Reverse the rank of a board game so that higher ranked board games have a lower value
    df_games["Rank:reversedboardgame"] = df_games["Rank:boardgame"].rank(ascending=False)

    # Save df_encoded as a pickle
    with open("df_games.pickle", "wb") as file:
        pickle.dump(df_games, file)

    # LLM section
    df_llm = df_games
    df_llm = df_llm[df_llm["NumUserRatings"] > 1000]
    # Make sure the games have a 7 rating or above
    df_llm = df_llm[df_llm["AvgRating"] > 7]
    df_llm = df_llm[["BGGId", "Name", "AvgRating", "NumUserRatings"]]
    df_llm.to_csv("df_games_llm.txt", index=False)

    # Loading in the mechanics data
    df_mechanics = pd.read_csv("./raw_data/mechanics.csv")
    # Make sure that the board games in mechanics.csv only include 
    # board games in games.csv
    df_games = df_games.set_index("BGGId")
    df_mechanics = df_mechanics.set_index("BGGId")
    # After the join we want the mechanics dataframe to have mechanics columns
    mechanics_columns = df_mechanics.columns
    df_mechanics = df_mechanics.join(df_games, how="inner")
    df_mechanics = df_mechanics[mechanics_columns]
    df_mechanics.reset_index(inplace=True)
    df_games.reset_index(inplace=True)

    with open("df_mechanics.pickle", "wb") as file:
        pickle.dump(df_mechanics, file)
    print(df_games.shape)
    print(df_mechanics.shape)

if __name__ == "__main__":
    main()