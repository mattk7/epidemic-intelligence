#!/usr/bin/env python
# coding: utf-8

# In[45]:


def generate_continuous_bins(category, min_age, max_age):
    """
    Generates the SQL snippet for a given category and continuous age range.
    """
    age_bins = []
    age_intervals = [(0, 4), (5, 8), (9, 12), (13, 17), (18, 23), (24, 29), (30, 34), (35, 39),
                     (40, 44), (45, 49), (50, 54), (55, 59), (60, 64), (65, 69), (70, 74), (75, 'plus')]

    for start, end in age_intervals:
        try:
            if start >= min_age and (end <= max_age):
                age_bins.append(f"{category}_{start}_{end}")
        except(TypeError):
            if start <= max_age or max_age == 'plus':
                age_bins.append(f"{category}_{start}_{end}")
    
    return age_bins


def build_country_query(table, country_ids, run_ids, min_age, max_age, categories, grouped=True):
    """
    Builds the full SQL query for multiple categories, country_ids, and run_ids with continuous age range.
    
    Args:
    - table (str): name of table to be queried in `project.dataset.table` formtable, 
    - country_ids (list): List of country identifiers.
    - run_ids (list): List of run identifiers.
    - min_age (int): The minimum age for the age range.
    - max_age (int or str): The maximum age for the age range (can be 'plus').
    - categories (list): List of categories to generate the SQL for (e.g., ['Susceptible', 'Infectious']).
    - grouped (bool): Whether to sum the age bins for each category or return them separately.
    
    Returns:
    - str: Full SQL query for the categories and age ranges.
    """
    category_statements = []
    group_by_columns = ['date', 'country_id', 'run_id']  # Common columns to always group by
    
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

    # Build the WHERE clause for country_ids and run_ids
    country_ids_str = ', '.join(map(str, country_ids))
    run_ids_str = ', '.join(map(str, run_ids))
    
    # Construct the SQL query
    sql_query = f"""
    SELECT
        date,
        country_id,
        run_id,
        {', '.join(category_statements)}
    FROM `{table}`
    WHERE country_id IN ({country_ids_str}) AND run_id IN ({run_ids_str})
    GROUP BY {', '.join(group_by_columns)}
    ORDER BY date;
    """
    
    return sql_query.replace("\n", " ").strip()

# Example usage
table = 'net-data-viz-handbook.sri_data.SIR_0_countries_incidence_daily'
country_ids = [215, 216]
run_ids = [1, 2]
min_age = 5
max_age = 50
categories = ['Susceptible', 'Infectious']

# Generate SQL query with grouped=True (default behavior)
query_grouped = build_country_query(table, country_ids, run_ids, min_age, max_age, categories, grouped=True)

# Generate SQL query with grouped=False (return individual bins)
query_separate = build_country_query(table, country_ids, run_ids, min_age, max_age, categories, grouped=False)

query_grouped, query_separate

