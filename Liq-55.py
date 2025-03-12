import boto3 #need to install in the environment
import weaviate #need to install in the environment
import json
import re
import concurrent.futures

# Set up AWS Bedrock client (update with the correct AWS region)
bedrock = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')

# Set up Weaviate client (update with your Weaviate instance URL)
client = weaviate.Client("http://localhost:8080")

# Get user input for search request and result limit
user_input = input("Enter your search request: ")
limit_input = input("Enter the number of results to retrieve (default: 10): ")
limit = int(limit_input) if limit_input.isdigit() else 10  # Default to 10 if input is empty or invalid


# Function to sanitize extracted parameters
def sanitize_input(value):
    if isinstance(value, str):
        sanitized_value = re.sub(r"[^a-zA-Z0-9 .,-]", "", value).strip()
        return sanitized_value if sanitized_value else "N/A"
    return value


# Function to extract query, year, and jurisdiction using AWS Bedrock
def extract_search_parameters(user_input):
    prompt = f"""
    Extract the search parameters from the following user request:
    "{user_input}"

    Return a JSON object with:
    - "query": The type of legal case being searched (e.g., "criminal cases").
    - "year": The year of the cases (if mentioned).
    - "jurisdiction": The location of the court (if mentioned).

    Example output format:
    {{
        "query": "criminal cases",
        "year": 2017,
        "jurisdiction": "California local courts"
    }}
    """
    try:
        response = bedrock.invoke_model(
            modelId="anthropic.claude-v2",  # Update with the correct model ID
            contentType="application/json",
            accept="application/json",
            body=json.dumps({"prompt": prompt, "max_tokens": 200})
        )
        response_body = json.loads(response["body"].read().decode("utf-8"))
        extracted_params = json.loads(response_body.get("completion", "{}"))

        # Sanitize extracted parameters
        return {
            "query": sanitize_input(extracted_params.get("query", "N/A")),
            "year": sanitize_input(extracted_params.get("year", None)),
            "jurisdiction": sanitize_input(extracted_params.get("jurisdiction", "N/A"))
        }
    except (KeyError, json.JSONDecodeError, IndexError):
        return {"query": "N/A", "year": None, "jurisdiction": "N/A"}


# Extract query parameters
parameters = extract_search_parameters(user_input)
query = parameters.get("query", "N/A")
year = parameters.get("year", None)
jurisdiction = parameters.get("jurisdiction", "N/A")

# Define Weaviate filters based on extracted parameters
filters = {"path": ["year"], "operator": "Equal", "valueInt": year} if year else None
filters_location = {"path": ["jurisdiction"], "operator": "Equal",
                    "valueString": jurisdiction} if jurisdiction != "N/A" else None

# Build the Weaviate hybrid search query
query_builder = client.query.get(
    "LegalCase", ["title", "summary", "date", "judge", "jurisdiction"]
).with_hybrid(query=query).with_limit(limit)

# Apply filters if available
if filters:
    query_builder = query_builder.with_where(filters)
if filters_location:
    query_builder = query_builder.with_where(filters_location)

# Execute the query against Weaviate
try:
    response = query_builder.do()
    cases = response.get("data", {}).get("Get", {}).get("LegalCase", [])
except AttributeError:
    cases = []


# Function to generate a short summary using AWS Bedrock
def generate_summary(text):
    prompt = f"Summarize this legal case in two sentences:\n{text}"
    try:
        response = bedrock.invoke_model(
            modelId="anthropic.claude-v2",  # Update with the correct model ID
            contentType="application/json",
            accept="application/json",
            body=json.dumps({"prompt": prompt, "max_tokens": 100})
        )
        response_body = json.loads(response["body"].read().decode("utf-8"))
        return response_body.get("completion", "Summary unavailable.").strip()
    except (KeyError, IndexError):
        return "Summary unavailable."


# Use ThreadPoolExecutor to concurrently generate summaries for each case
with concurrent.futures.ThreadPoolExecutor() as executor:
    # Submit summary generation tasks for each case
    future_to_case = {
        executor.submit(generate_summary, case.get("summary", "No summary available.")): case
        for case in cases
    }
    # Process the completed futures as they finish
    for future in concurrent.futures.as_completed(future_to_case):
        case = future_to_case[future]
        try:
            case["short_summary"] = future.result()
        except Exception:
            case["short_summary"] = "Summary unavailable."

# Output final structured results
print(json.dumps(cases, indent=2))
