import subprocess
import pandas as pd
import os
import time

# Define the CSV file path
csv_file = "game_results.csv"

# Check if the CSV file exists
if os.path.exists(csv_file):
    # Read the existing CSV file
    existing_df = pd.read_csv(csv_file)
else:
    # Create an empty DataFrame with the required columns if the file does not exist
    existing_df = pd.DataFrame(columns=["size", "time","wins", "games_played", "percent_wins"])

# Ensure the DataFrame has a row for the current game size
def help(size, time):
    global existing_df  # Declare existing_df as global
    new_row = pd.DataFrame([{"size": size, "time": time, "wins": 0, "games_played": 0, "percent_wins": 0}])
    existing_df = pd.concat([existing_df, new_row], ignore_index=True)


def analyze_game(iterations = 1000, size = 5, time = 20):
    global existing_df  # Declare existing_df as global
    command = f"python3 game.py ai random --dim {size} --time {time} --mode server"
    help(size, time)
    win = 0
    # Run the command 1000 times
    for iter in range(iterations):
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        winner = 0
        try:
            winner = int(result.stdout)
        except:
            winner = 0
        print(f"size: {size}, winner: {winner}, iter: {iter}")
        
        # Update the DataFrame with the new result
        if winner == 1:
            win += 1
    existing_df.loc[existing_df["size"] == size, "wins"] += win
    existing_df.loc[existing_df["size"] == size, "games_played"] += iterations
        
def update_csv():
    global existing_df  # Declare existing_df as global
    for index, row in existing_df.iterrows():
        if row["games_played"] > 0:
            existing_df.at[index, "percent_wins"] = row["wins"] / row["games_played"]
        else:
            existing_df.at[index, "percent_wins"] = 0
    existing_df.to_csv(csv_file, index=False)

start = time.time()
for i in range(11, 13):
    for j in range(3*i, 3*i+1):
        analyze_game(iterations=100, size=i, time=j)
        # Save the updated DataFrame to the CSV file after each run
    update_csv()
end = time.time()
print(end - start)

