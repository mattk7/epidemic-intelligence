def execute(client, query):
    # Execute query
    query_job = client.query(query)

    # Convert result to a Pandas DataFrame
    result_df = query_job.to_dataframe()

    return result_df