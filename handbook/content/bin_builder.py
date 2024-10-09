def build_country_query(table, country_ids, run_ids, min_age, max_age, categories, grouped=True):
    """
    Builds the full SQL query for multiple categories, country_ids, and run_ids with continuous age range.
    
    Args:
    - table (str): name of table to be queried in `project.dataset.table` formtable, 
    - country_ids (list): List of country identifiers.
    - run_ids (list or str): List of run identifiers or 'all' to include all runs.
    - min_age (int): The minimum age for the age range.
    - max_age (int or str): The maximum age for the age range (can be 'plus').
    - categories (list): List of categories to generate the SQL for (e.g., ['Susceptible', 'Infectious']).
    - grouped (bool): Whether to sum the age bins for each category or return them separately.
    
    Returns:
    - str: Full SQL query for the categories and age ranges.
    """
    category_statements = []
    group_by_columns = ['date', 'country_id']  # Common columns to always group by
    
    # Generate the SQL statements for each category
    for category in categories:
        category_bins = generate_continuous_bins(category, min_age, max_age)
        if grouped:
            # If grouped, sum all the bins
            category_statements.append(f"SUM({' + '.join(category_bins)}) AS total_{category.lower()}")
        else:
            # If not grouped, return each bin separately and add to GROUP BY
            for bin_ in category_bins:
                category_statements.append(f"{bin_} AS {bin_.lower()}")
                group_by_columns.append(bin_)

    # Build the WHERE clause for country_ids
    country_ids_str = ', '.join(map(str, country_ids))
    where_clause = f"country_id IN ({country_ids_str})"
    
    # Only filter by run_id if run_ids is not 'all'
    if run_ids != 'all':
        run_ids_str = ', '.join(map(str, run_ids))
        where_clause += f" AND run_id IN ({run_ids_str})"
        group_by_columns.append('run_id')

    # Construct the SQL query
    sql_query = f"""
    SELECT
        date,
        country_id,
        {', '.join(group_by_columns)}
        {', '.join(category_statements)}
    FROM `{table}`
    WHERE {where_clause}
    GROUP BY {', '.join(group_by_columns)}
    ORDER BY date;
    """
    
    return sql_query.replace("\n", " ").strip()

# Example usage
table = 'net-data-viz-handbook.sri_data.SIR_0_countries_incidence_daily'
country_ids = [215, 216]
run_ids = 'all'
min_age = 5
max_age = 50
categories = ['Susceptible', 'Infectious']

# Generate SQL query with grouped=True (default behavior)
query_grouped = build_country_query(table, country_ids, run_ids, min_age, max_age, categories, grouped=True)

# Generate SQL query with grouped=False (return individual bins)
query_separate = build_country_query(table, country_ids, run_ids, min_age, max_age, categories, grouped=False)

query_grouped