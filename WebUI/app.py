from flask import Flask, render_template, request, redirect, url_for, session
import numpy as np
import pandas as pd
import sys
import pickle
import os
import ast
sys.path.insert(0, "./app_utils/")
from contentbased import Profile
from llmbased import get_recommendations_better

app = Flask(__name__)
app.secret_key = b"it's a secret"

@app.route("/", methods=["GET", "POST"])
def index():
	return redirect(url_for("home"))

@app.route("/home", methods=["GET", "POST"])
def home():
	session.clear()
	# Remove the user_games pickle
	if (os.path.exists("user_games.pkl")):
		os.remove("user_games.pkl")
	return render_template("home.html")

@app.route("/machine_learning", methods=["GET", "POST"])
def machine_learning():
	recommendations = session.get("recommendations", {})
	llm_recommendations = session.get("llm_recommendations", {})
	n_recommendations = session.get("n_recommendations", "10")
	n_recommendations_llm = session.get("n_recommendations", "10")
	if (request.method == "POST"):
		if ("num_recs_content" in request.form):
			recommendations = {} # Clear the dictionary to add new recommendations
			n_recommendations = request.form.get("num_recs_content")
			session["n_recommendations"] = n_recommendations
			with open("model.pkl", "rb") as file:
				model = pickle.load(file)
			try:
				with open("user_games.pkl", "rb") as file:
					user_games = pickle.load(file)
			except FileNotFoundError:
				recommendations = {}
				session["recommendations"] = {}
				return render_template("machine_learning.html", recommendations=recommendations, 
							n_recommendations=n_recommendations, llm_recommendations=llm_recommendations, 
							n_recommendations_llm=n_recommendations_llm)
			bgg_ids = list(map(int, list(user_games.keys())))
			model_input = dict(zip(bgg_ids, [10 for i in range(len(user_games))]))
			try:
				model.fit_recommendations(model_input)
			# If model_input is empty
			except ValueError:
				# Do nothing
				return render_template("machine_learning.html", recommendations=recommendations, 
							n_recommendations=n_recommendations, llm_recommendations=llm_recommendations, 
							n_recommendations_llm=n_recommendations_llm)
			model_results = model.find_recommendations(int(n_recommendations))
			# Turning model_results into a nice, table-friendly format
			for index, row in model_results.iterrows():
				recommendations[row["BGGId"]] = [str(row["Name"]), round(float(row["AvgRating"]), 2), 
												int(row["NumUserRatings"]), int(row["YearPublished"]), 
												int(row["BGGId"]), float(row["GameWeight"])]
			session["recommendations"] = recommendations
		if ("num_recs_llm" in request.form):
			llm_recommendations = {}
			n_recommendations_llm = request.form.get("num_recs_llm")
			session["n_recommendations_llm"] = n_recommendations_llm
			try:
				with open("user_games.pkl", "rb") as file:
					user_games = pickle.load(file)
			except FileNotFoundError:
				llm_recommendations = {}
				session["llm_recommendations"] = {}
				return render_template("machine_learning.html", recommendations=recommendations, 
							n_recommendations=n_recommendations, llm_recommendations=llm_recommendations, 
							n_recommendations_llm=n_recommendations_llm)
			bgg_ids = list(map(int, list(user_games.keys())))
			model_input = dict(zip(bgg_ids, [10 for i in range(len(user_games))]))
			llm_recommendations = get_recommendations_better(model_input, "df_games_llm.txt", n_recommendations_llm)
			session["llm_recommendations"] = llm_recommendations		
	return render_template("machine_learning.html", recommendations=recommendations, 
							n_recommendations=n_recommendations, llm_recommendations=llm_recommendations, 
							n_recommendations_llm=n_recommendations_llm)

@app.route("/profile", methods=["GET", "POST"])
def profile():
	# Load in the saved user_games variable
	try:
		with open("user_games.pkl", "rb") as file:
			user_games = pickle.load(file)
	except: # If we can't open the file, create a new user_games dictionary
		user_games = {}
	# search_results follows this format: BGGId: [Name, AvgRating, NumUserRatings, YearPublished]
	search_results = session.get("search_results", {})
	search_query = session.get("search_query", "")
	if (request.method == "POST"):
		if ("search_field" in request.form and request.form.get("search_field") != ""):
			search_results = {}
			search_query = request.form.get("search_field")
			session["search_query"] = search_query
			# Load in df_games and get the search results
			df_games = pd.read_pickle("./app_utils/df_games.pickle")
			df_filtered = df_games[df_games["Name"].str.contains(search_query, case=False)]
			# Sort df_filtered by number of reviews to get the most relevant results TODO, broken
			# df_filtered = df_filtered.sort_values(by="NumUserRatings", ascending=False)
			# Filter out games that the user has already added from df_filtered TODO, not great
			# df_filtered = df_filtered[~df_filtered["BGGId"].isin(user_games.keys())]
			# Loop over df_filtered and add the results to search_results
			for index, row in df_filtered.iterrows():
				search_results[row["BGGId"]] = [str(row["Name"]), round(float(row["AvgRating"]), 2), 
												int(row["NumUserRatings"]), int(row["YearPublished"]), 
												int(row["BGGId"])]
				session["search_results"] = search_results
		# Add a new game to the favorites list user_games
		button_output_add = request.form.get("action_add")
		if (button_output_add != None):
			button_output_add = button_output_add.split(", ")
			user_games[int(button_output_add[0])] = button_output_add[1]
		
		# Remove a board game from the favorites list
		button_output_remove = request.form.get("action_remove")
		if (button_output_remove != None):
			del user_games[int(button_output_remove)]

	# Save user_games as a pickle
	with open("user_games.pkl", "wb") as file:
		pickle.dump(user_games, file)

	return render_template("profile.html", user_games=user_games, search_results=search_results, 
										   search_query=search_query)

if __name__ == "__main__":
	app.run(host="0.0.0.0", port=8080)