CREATE TABLE poke_types (
    type_name TEXT NOT NULL,
    pokemon_id INTEGER NOT NULL
);

CREATE TABLE poke_most_abilities (
	ability_name TEXT NOT NULL,
	moves INTEGER NOT NULL,
);

CREATE TABLE poke_avg_experience_per_type (
	type_name TEXT NOT NULL,
	avg_experience INTEGER NOT NULL,
);

CREATE TABLE poke_types_dmg_relations (
	pokemon_id INTEGER NOT NULL,
	type_name TEXT NOT NULL,
	score FLOAT NOT NULL,
);