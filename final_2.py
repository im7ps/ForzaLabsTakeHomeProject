import pokebase as pb
import requests
import pandas as pd
import sqlite3
import numpy as np
from typing import List, Dict, Optional

def count_max_abilities(generation_id: int, max: Optional[int]= -1) -> int:
	"""
	Count the maximum number of abilities across all Pokémon species in a given generation.

	Args:
		generation_id: The ID of the Pokémon generation to process.
		max: The initial maximum value to compare against. Defaults to -1.

	Returns:
		int: The maximum number of abilities found across the Pokémon species in the specified generation.
	"""
	gen = pb.generation(generation_id)

	for specie in gen.pokemon_species:
		abilities_url = f"https://pokeapi.co/api/v2/pokemon/{specie.name}"
		pokemon_data = requests.get(abilities_url).json()
		temp = len(pokemon_data["abilities"])
		if temp > max:
			max = temp
	return max


def get_poke_base_info() -> pd.DataFrame:
	"""
	Creates a df that stores pokemon base data like id, name, weight, height and base experience.
	Returns:
		A DataFrame with the base information for every pokemon in the first generation.
	"""
	data = []

	for i in range(1, 152):
		try:
			response = requests.get(f"https://pokeapi.co/api/v2/pokemon/{i}")
			response.raise_for_status()
			pokemon = response.json()

			data.append({
				"name": pokemon["name"],
				"height": pokemon["height"],
				"weight": pokemon["weight"],
				"base_experience": pokemon["base_experience"],
			})
		except requests.exceptions.HTTPError as e:
			print(f"get_poke_base_info error on {i}: {e}")

	data = pd.DataFrame(data)
	data['id'] = data.index + 1
 
	data.drop_duplicates(inplace=True)

	return data


def get_poke_types(base_url) -> pd.DataFrame:
	"""
	Creates a df that has the pokemon type name and a string with the list of pokemons who have that type.
	Returns:
		A DataFrame that has the pokemon type name and a string with the list of pokemons who have that type..
	"""
	type_id_lists = {}
	generation_url = f"{base_url}generation/1/"

	try:
		response = requests.get(generation_url)
		response.raise_for_status()
	except requests.exceptions.HTTPError as e:
		print(f"get_poke_types error: {e}")
		return pd.DataFrame()

	generation_data = response.json()

	type_urls = [type_data['url'] for type_data in generation_data['types']]

	for type_url in type_urls:
		try:
			type_response = requests.get(type_url)
		except requests.exceptions.HTTPError as e:
			print(f"type_response error: {e}")
			continue
		finally:
			type_data = type_response.json()

		type_name = type_data["name"]
		type_id_lists[type_name] = []


		for pokemon in type_data['pokemon']:
			poke_url = pokemon['pokemon']['url']
			poke_id = int(poke_url.rstrip('/').split('/')[-1])
			if 1 <= poke_id <= 151:
				type_id_lists[type_name].append(poke_id)
		
	type_to_ids = pd.DataFrame([(key, value) for key, value in type_id_lists.items()], columns=["poke_type", "id"])
	return type_to_ids



def expand_pokemon(df) -> pd.DataFrame:
	"""
	Takes a df and explode his rows, in this particular case i do some string manipulation because i have a string and not a list
	Returns:
		A DataFrame that now as a list instead of a string under the column id
	"""
	expanded_rows = []
	for _, row in df.iterrows():
		type_name = row["poke_type"]
		pokemon_ids = str(row["id"]).replace("[", "").replace("]", "").split(",")
		for pokemon_id in pokemon_ids:
			expanded_rows.append({'type_name': type_name, 'pokemon_id': int(pokemon_id)})
	return pd.DataFrame(expanded_rows)


def sort_poke_types(df) -> pd.DataFrame:
	"""
	Fetch and sort Pokémon types by Pokémon ID and type name.

	Returns:
		A sorted DataFrame with Pokémon types expanded and sorted by 'pokemon_id' and 'type_name'.
	"""
	# Sort the DataFrame by 'pokemon_id' and 'type_name' in ascending order
	return df.sort_values(by=['pokemon_id', 'type_name'], ascending=True).reset_index(drop=True)


def get_all_poke_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process Pokémon type data to produce a DataFrame with unique combined types and their representative Pokémon IDs.

    Returns:
        A DataFrame with columns 'combined_types' and 'id', where a combined type is for example grasspoison, the id is the id of that pokemon
    """
    # Combine types into a single string for each Pokémon, sorted to ensure uniqueness
    pokemon_types: Dict[int, str] = (
        df.groupby('pokemon_id')['type_name']
        .agg(''.join)  # Aggregate by joining the type names into a single string
        .to_dict()
    )
    
    # Find unique type combinations and the corresponding Pokémon ID
    unique_types_to_id = {}
    for pokemon_id, combined_types in pokemon_types.items():
        if combined_types not in unique_types_to_id:
            unique_types_to_id[combined_types] = pokemon_id

    # Create a DataFrame with the results
    data = {
        "combined_types": list(unique_types_to_id.keys()),
        "id": list(unique_types_to_id.values())
    }

    return pd.DataFrame(data)


def fetch_damage_relations() -> pd.DataFrame:
	"""
	Fetch damage relations for first-generation Pokémon from the PokéAPI and return a DataFrame.

	Returns:
		A DataFrame containing damage relations for each Pokémon type to the others.
	"""

	base_url = "https://pokeapi.co/api/v2/pokemon/"
	type_url = "https://pokeapi.co/api/v2/type/"

	first_gen_pokemon_ids = range(1, 152)

	# Lista per memorizzare i risultati
	damage_relations_data: List[Dict] = []

	for pokemon_id in first_gen_pokemon_ids:
		try:
			response = requests.get(f"{base_url}{pokemon_id}")
			response.raise_for_status()
			pokemon_data = response.json()

			# Ottieni i tipi del Pokémon
			type_names = [t['type']['name'] for t in pokemon_data['types']]

			# Inizializza i set per i dati di relazione dei danni
			no_damage_to = list()
			half_damage_to = list()
			double_damage_to = list()
			no_damage_from = list()
			half_damage_from = list()
			double_damage_from = list()

			# Itera attraverso i tipi per ottenere le relazioni
			for type_name in type_names:
				type_response = requests.get(f"{type_url}{type_name}")
				if type_response.status_code == 200:
					type_data = type_response.json()['damage_relations']

					no_damage_to += [t['name'] for t in type_data['no_damage_to']]
					half_damage_to += [t['name'] for t in type_data['half_damage_to']]
					double_damage_to += [t['name'] for t in type_data['double_damage_to']]
					no_damage_from += [t['name'] for t in type_data['no_damage_from']]
					half_damage_from += [t['name'] for t in type_data['half_damage_from']]
					double_damage_from += [t['name'] for t in type_data['double_damage_from']]

			# Aggiungi il risultato alla lista
			damage_relations_data.append({
				"id": pokemon_id,
				"no_damage_to": no_damage_to,
				"half_damage_to": half_damage_to,
				"double_damage_to": double_damage_to,
				"no_damage_from": no_damage_from,
				"half_damage_from": half_damage_from,
				"double_damage_from": double_damage_from,
			})
		except requests.exceptions.HTTPError as e:
			print(f"fetch_damage_relations error on id: {pokemon_id}\n{e}")

	return pd.DataFrame(damage_relations_data)


def create_dmg_relations_csv(poke_types_all: pd.DataFrame, damage_relations: pd.DataFrame) -> pd.DataFrame:
	"""
	Merge Pokémon type data with damage relations and return the final DataFrame.

	Args:
		poke_types_all: DataFrame containing Pokémon type information, 
									   including 'id' and 'combined_types'.
		damage_relations: DataFrame containing Pokémon's type damage relations but not the pokemon type.

	Returns:
		A DataFrame containing damage relations merged with Pokémon types,
					  including columns for no/half/double damage relations.
	"""
	# Define columns to merge and validate their presence
	required_columns = [
		"id", "no_damage_to", "half_damage_to", "double_damage_to",
		"no_damage_from", "half_damage_from", "double_damage_from"
	]
	missing_columns = [col for col in required_columns if col not in damage_relations.columns]
	if missing_columns:
		raise ValueError(f"Missing required columns in damage relations data: {missing_columns}")

	# Filter relevant columns from damage_relations
	damage_relations = damage_relations[required_columns]

	# Validate 'id' column in poke_types_all
	if "id" not in poke_types_all.columns:
		raise ValueError("'poke_types_all' DataFrame must contain an 'id' column.")

	# Merge DataFrames
	merged_df = pd.merge(
		poke_types_all,
		damage_relations,
		how="inner",
		on="id"
	)

	# Select final columns
	final_columns = [
		"id", "combined_types", "no_damage_to", "half_damage_to", "double_damage_to",
		"no_damage_from", "half_damage_from", "double_damage_from"
	]
	missing_final_columns = [col for col in final_columns if col not in merged_df.columns]
	if missing_final_columns:
		raise ValueError(f"Missing columns after merge: {missing_final_columns}")

	return merged_df[final_columns]


def get_poke_types_all_gen(base_url) -> pd.DataFrame:
	"""
	Fetch Pokémon types and their counts from the API and return them as a DataFrame.

	ATTENTION: In calculating damage relations I decided to assign the stellar type to all the Pokemon because by using an item each Pokemon can become stellar type in combat
			the way i calculate the stellar is 
	Returns:
		 A DataFrame with columns ["poke_type", "poke_num"] representing Pokémon types and the number of Pokémon for each type.
	"""

	type_counts = []
	types_url = f"{base_url}type/"
	try:
		response = requests.get(types_url)
		response.raise_for_status()
		types_data = response.json()
	except requests.exceptions.HTTPError as e:
		print(f"get_poke_types_all_gen error on {e}")

	type_urls = [type_data['url'] for type_data in types_data['results']]
	total = 0

	for type_url in type_urls:
		try:
			response = requests.get(type_url)
			response.raise_for_status()
			type_data = response.json()
			type_name = type_data['name']
			pokemon_count = len(type_data['pokemon'])
			type_counts.append({"poke_type": type_name, "poke_num": pokemon_count})
			total += pokemon_count
		except requests.exceptions.HTTPError as e:
			print(f"get_poke_types_all_gen error on {e}")
			continue
	stellar_entry = next((entry for entry in type_counts if entry["poke_type"] == "stellar"), None)
	if stellar_entry:
		stellar_entry["poke_num"] = total
	return pd.DataFrame(type_counts)


def get_score(s_dict: dict, s_list: list, s_multiplier: float) -> float:
	"""
	Calculate the score for each type based on the damage relations of that type multiplied by an hardcoded factor
	Multitypes are handled this way:
		the score of the two types taken individually is added, 
		in this way i think i track both the quadruple-damage to the types vulnerable to both and the immunity to a type even if only one of your types is immune to that (Pokemon rules)
	
	Args:
		s_dict: Dizionario contenente chiavi nella forma "{type}_percentage".
		s_list: Lista di tipi (stringhe) per cui calcolare lo score.
		s_multiplier: Valore moltiplicativo per lo score calcolato.

	"""
	score = 0
	if s_list:
		valid_types = [ptype for ptype in s_list if f"{ptype}_percentage" in s_dict]
		percentages = np.array([s_dict[f"{ptype}_percentage"] for ptype in valid_types])
		score = percentages.sum() * s_multiplier
	return score


def types_get_popularity_all_gen(types_csv):
	"""
	Creates a df that represents the percetage of presence of a type across all generations of pokemons,
	used to calculate the score for the damage relations query
	"""
 
	# here the / 2 is caused by the stellar type having the total of the pokemon as number of stellar pokemons
	total = types_csv["poke_num"].astype(int).sum() / 2
	if total == 0:
		return None
		
	d = {}
		
	for row in types_csv.itertuples(index=False):
		popularity = row.poke_num / total
		d[f"{row.poke_type}_percentage"] = round(popularity, 2)

	return d

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def query_pokemon_most_abilities(max, base_url):
	"""
	Produces an sql table filled with the pokemon in the first generation with the most number of abilities.
	Ties are handled this way:
		all the pokemons with the most number of abilities are considered as first.

	ATTENTION: if you wanted the pokemon with the most moves the pokemon is mew with 375 moves, the code is almost the same but with the "moves" parameter we have a clean winner
				the code is not included because i feel it's not part of the assignment
	"""
	df_results = []
	# Scanning through all the first generation pokemons getting the length of thier "abilities" list, if it's the max the pokemon gets added
	for i in range(1, 152):
		try:
			response = requests.get(base_url + "pokemon/" + str(i), timeout=5)
			response.raise_for_status()
			pokemon = response.json()

			if len(pokemon["abilities"]) == max:
				df_results.append({"pokemon": pokemon["name"], "abilities": max})
		except requests.exceptions.HTTPError as e:
			print(f"query_pokemon_most_abilities error on {i}: {e}")

	if not df_results:
		print("query_pokemon_most_abilities failed no df_results")
		return

	df_results = pd.DataFrame(df_results, columns=["pokemon", "abilities"])

	try:
		conn = sqlite3.connect("pokemon_most_abilities.db")
		df_results.to_sql("poke_most_abilities", conn, if_exists="replace", index=False)
		query = """SELECT * FROM poke_most_abilities"""
		df_results = pd.read_sql(query, conn)
	except:
		print(f"query_pokemon_most_abilities failed no sql conn")
	finally:
		conn.close()
		return df_results



def query_avg_experience_per_type(info_base: pd.DataFrame, types: pd.DataFrame):
	"""
	Produces an sql table filled with the average base experience for each pokemon in the first generation.
	Multitypes are handled this way:
 		The base experience of a pokemon of type grass-poison is added to both the grass and poison type,
		the pokemon counts for both as well.
	"""
	# Creation of the df with the pokemon base info and their type
	info_base_required = info_base[["id", "base_experience"]]
	merged = pd.merge(types, info_base_required, left_on="pokemon_id", right_on="id", how="left")

	# Calculates the mean of the selected column by pokemon types and creates the df
	avg_exp = merged.groupby("type_name")["base_experience"].mean().round(2).reset_index()
	avg_exp.columns = ['type', 'avg_base_experience']

	conn = sqlite3.connect("poke_avg_experience_per_type.db")
	avg_exp.to_sql("poke_avg_experience_per_type", conn, if_exists="replace", index=False)
	query = "SELECT * FROM poke_avg_experience_per_type"
	df = pd.read_sql(query, conn)
	conn.close()
	return df



def query_weight_top5_per_type(info_base: pd.DataFrame, types: pd.DataFrame):
	"""
	Produces a DataFrame filled with the heaviest 5 Pokémon for each type in the first generation.
	Multitypes are handled this way:
 		since a pokemon grass-poison can be the heaviest in the grass group and in the poison group i decided to make them count for both,
		so a pokemon grass-poison will be 1st in the grass group and in the poison group, the grasspoison group is not considered.
	"""
	info_base_selected = info_base[["id", "name", "weight"]]
	types_selected = types[["pokemon_id", "type_name"]]

	# Creation of df ordered by weight and creation of column rank
	merged = pd.merge(info_base_selected, types_selected, left_on="id", right_on="pokemon_id")
	merged["rank"] = merged.groupby("type_name")["weight"].rank(method="first", ascending=False)

	# Selection of the first 5 by the rank column
	top5_per_type = merged[merged["rank"] <= 5].sort_values(by=["type_name", "rank"])
	top5_per_type = top5_per_type.drop(columns=["rank"])
	
	# Transformation of the df to improve readability and further manipulations
	df = (
		top5_per_type.assign(rank=top5_per_type.groupby("type_name").cumcount() + 1)
		.pivot(index="type_name", columns="rank", values="name")
	)
	df.columns = [f"Top {col}" for col in df.columns]
	df.reset_index(inplace=True)
	df.rename(columns={"type_name": "poke_type"}, inplace=True)

	conn = sqlite3.connect("poke_weight_per_type.db")
	df.to_sql("poke_most_moves", conn, if_exists="replace", index=False)
	query = """SELECT * FROM poke_most_moves"""
	df_read = pd.read_sql_query(query, conn)
	conn.close()
	return df_read


def query_type_dmg_relations(all_types: pd.DataFrame, dmg_relations: pd.DataFrame, d: pd.DataFrame):
	"""
	Computes damage relations scores for Pokémon types and exports the results to a CSV file.
	The score is calculated based on the sum of the damage relations of the pokemon's type multiplied by a constant, for each type.
	"""

	# Damage multiplier dictionary, i think that being immune to something is far stronger than doing double damage, so the weight on the immune relations is higher
	d_multiplier = {
		"no_damage_to": -10,
		"half_damage_to": -2,
		"double_damage_to": 2,
		"no_damage_from": 10,
		"half_damage_from": 2,
		"double_damage_from": -2,
	}
	
	# Calculate how a pokemon type relates to the others based on the relations of each of his types and the popularity of the types he affects.
	def calculate_scores(column, multiplier):
		return dmg_relations[column].apply(lambda types: get_score(d, types, multiplier))


	score = sum(
		calculate_scores(column, multiplier)
		for column, multiplier in d_multiplier.items()
	)

	score_list = pd.DataFrame({"score": score.round(2)})
	final_df = pd.concat([all_types.reset_index(drop=True), score_list], axis=1)
	final_df = final_df.sort_values(by="score", ascending=False)

	conn = sqlite3.connect("poke_types_dmg_relations.db")
	final_df.to_sql("poke_types_dmg_relations", conn, if_exists="replace", index=False)
	query = """SELECT * FROM poke_types_dmg_relations"""
	final = pd.read_sql_query(query, conn)
	conn.close()
	return final

def main():
	base_url = "https://pokeapi.co/api/v2/"
	max_abilities: int = count_max_abilities(1)

	df_poke_base_info: pd.DataFrame = get_poke_base_info()

	df_poke_types: pd.DataFrame = get_poke_types(base_url)
	df_poke_types_exploded: pd.DataFrame = expand_pokemon(df_poke_types)

	df_poke_types_sorted: pd.DataFrame = sort_poke_types(df_poke_types_exploded)
	df_poke_types_all: pd.DataFrame = get_all_poke_types(df_poke_types_sorted)
	df_poke_types_all_gen: pd.DataFrame = get_poke_types_all_gen(base_url)

	df_damage_relations: pd.DataFrame = fetch_damage_relations()
	df_damage_relations_all_types: pd.DataFrame = create_dmg_relations_csv(df_poke_types_all, df_damage_relations)
	df_types_popularity_all_gen: pd.DataFrame = types_get_popularity_all_gen(df_poke_types_all_gen)


	query_pokemon_most_abilities(max_abilities, base_url)
	query_avg_experience_per_type(df_poke_base_info, df_poke_types_exploded)
	query_weight_top5_per_type(df_poke_base_info, df_poke_types_exploded)
	query_type_dmg_relations(df_poke_types_all, df_damage_relations_all_types, df_types_popularity_all_gen)


if __name__ == "__main__":
    main()

 