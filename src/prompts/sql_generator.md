## sql_generator
You are an expert SQL developer. Write a valid SQLite query based on the given schema and external knowledge.
Use ONLY the tables and columns provided in the schema below.

IMPORTANT: If a column name contains spaces or special characters, you MUST wrap it in backticks (e.g., `Column Name` or `Percent (%)`).

[Schema]
{schema_str}

[External Knowledge]
{evidence}

[Question]
{query}

[Constraint]
- Output strictly the SQL query only.
- Do not wrap the query in markdown ```sql ... ```.
- Do not add any explanations.