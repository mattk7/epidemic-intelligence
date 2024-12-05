import hashlib
import secrets
from google.cloud import bigquery

def execute(client, query):
    # Execute query
    query_job = client.query(query)

    # Convert result to a Pandas DataFrame
    result_df = query_job.result().to_dataframe()

    return result_df

def create_dataset(client, dataset_name):
    # Define the dataset ID (project ID and dataset name)
    dataset_id = f"{client.project}.{dataset_name}"
    
    # Create a Dataset instance with the custom name
    dataset = bigquery.Dataset(dataset_id)

    # Use the client to create the dataset
    try:
        dataset = client.create_dataset(dataset, exists_ok=False) 
        print(f"Dataset `{dataset_id}` created.")
        return True
    except(Exception):
        print(f"Dataset `{dataset_id}` already exists. Any tables in it may be overwritten.")
        if input('Enter "CONFIRM" if you wish to proceed: ') == 'CONFIRM':
            return True
        else:
            print('Aborting process.')
            return False
        
def generate_random_hash():
    # Generate a random string
    random_string = secrets.token_bytes(32)
    
    # Create a SHA-256 hash of the random string
    hash_object = hashlib.sha256(random_string)
    random_hash = hash_object.hexdigest()
    
    return random_hash

def build_geographic_filter(geo_level: str, geo_values, alias: str = "g_target") -> str:
    """Builds a geographic filter based on the provided level and values."""
    if geo_values is not None:  # Only filter if geo_values is provided
        if isinstance(geo_values, list):
            if isinstance(geo_values[0], int):
                values = ', '.join(str(val) for val in geo_values)  # For INT64
                return f"{alias}.{geo_level} IN ({values})"
            elif isinstance(geo_values[0], str):
                values = ', '.join(f"'{val}'" for val in geo_values)  # For STRING
                return f"{alias}.{geo_level} IN ({values})"
        else:
            if isinstance(geo_values, int):
                return f"{alias}.{geo_level} = {geo_values}"
            elif isinstance(geo_values, str):
                return f"{alias}.{geo_level} = '{geo_values}'"
    return ""  # Return empty string if no filtering is needed

def build_categorical_filter(categories, category_col: str = 'category', alias: str = "g_target") -> str:
    if categories is not None:  # Only filter if geo_values is provided
        if isinstance(categories, list):
            if isinstance(categories[0], int):
                values = ', '.join(str(val) for val in categories)  # For INT64
                return f"{alias}.{category_col} IN ({values})"
            elif isinstance(categories[0], str):
                values = ', '.join(f"'{val}'" for val in categories)  # For STRING
                return f"{alias}.{category_col} IN ({values})"
        else:
            if isinstance(categories, int):
                return f"{alias}.{category_col} = {categories}"
            elif isinstance(categories, str):
                return f"{alias}.{category_col} = '{categories}'"
    return ""  # Return empty string if no filtering is needed

def hex_to_rgba(hex_code, alpha):
    # Remove the "#" if it's there
    hex_code = hex_code.lstrip('#')
    
    # Convert hex to RGB values
    r = int(hex_code[0:2], 16)
    g = int(hex_code[2:4], 16)
    b = int(hex_code[4:6], 16)
    
    # Create the RGBA string
    return f"rgba({r}, {g}, {b}, {alpha})"