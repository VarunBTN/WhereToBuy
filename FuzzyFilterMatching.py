import os
import json
from serpapi.google_search import GoogleSearch
from dotenv import load_dotenv
from pprint import pprint
from fuzzywuzzy import fuzz

load_dotenv()

# Load API key from .env file
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

def search_product(product_name, producer=None, varietal=None, vintage=None, location="London"):
    """Search for a product on Google Shopping using SerpAPI and return simplified results."""
     
    # Build query string
    query_parts = [product_name]
    if producer:
        query_parts.append(producer)
    if varietal:
        query_parts.append(varietal)
    if vintage:
        query_parts.append(str(vintage))
    
    query = " ".join(query_parts)
    
    params = {
        "engine": "google_shopping",
        "q": query,
        "hl": "en",
        "gl": "uk",  # Targeting UK results (adjust as needed)
        "api_key": SERPAPI_KEY
    }
    
    search = GoogleSearch(params)
    results = search.get_dict()
    
    # Extract relevant data
    extracted_results = []
    for item in results.get("shopping_results", []):
        product_info = {
            "product_name": item.get("title"),
            "store_name": item.get("source"),
            "price": item.get("price"),
            "rating": item.get("rating"),
            "link": item.get("link")  # Added link for potential future use
        }
        extracted_results.append(product_info)
    return extracted_results

def filter_with_fuzzy_matching(results, target_product, threshold=80):
    """
    Filter results using fuzzy matching and negative keyword exclusion
    """
    # List of negative keywords to exclude
    negative_keywords = [
        "empty", "bottle only", "decanted", "decant", "used",
        "vintage bottle", "old bottle", "collector", "display",
        "ornamental", "set of", "pack of", "lot of", "souvenir", "miniature"
    ]
    
    matches = []
    for item in results:
        product_name = item['product_name'].lower()
        
        # Skip if product contains any negative keywords
        if any(keyword in product_name for keyword in negative_keywords):
            continue
        
        # Calculate similarity score
        score = fuzz.token_set_ratio(target_product.lower(), product_name)
        
        # Check if score meets threshold
        if score >= threshold:
            item_with_score = item.copy()
            item_with_score['similarity_score'] = score
            matches.append(item_with_score)
    
    # Sort by score (highest first)
    matches.sort(key=lambda x: x['similarity_score'], reverse=True)
    return matches

def summarize_results(filtered_results):
    """Return the cheapest, highest-rated (if available), or best match results."""
    if not filtered_results:
        print("No matches found with the given criteria.")
        return

    # Helper: convert "£249.00" → 249.00
    def parse_price(p):
        if not p:
            return float("inf")
        return float(p.replace("£", "").replace(",", "").strip())

    # Cheapest result
    cheapest = min(filtered_results, key=lambda x: parse_price(x["price"]))
    print("\nCheapest:")
    print(f"Product: {cheapest['product_name']}")
    print(f"Price: {cheapest['price']}")
    print(f"Store: {cheapest['store_name']}")
    print(f"Rating: {cheapest['rating']}")

    # Highest rated result
    rated_results = [r for r in filtered_results if r["rating"] is not None]
    if rated_results:
        highest_rated = max(rated_results, key=lambda x: x["rating"])
        print("\nHighest rated:")
        print(f"Product: {highest_rated['product_name']}")
        print(f"Price: {highest_rated['price']}")
        print(f"Store: {highest_rated['store_name']}")
        print(f"Rating: {highest_rated['rating']}")
    else:
        # Fallback: show best match by similarity score
        best_match = max(filtered_results, key=lambda x: x["similarity_score"])
        print("\nBest match:")
        print(f"Product: {best_match['product_name']}")
        print(f"Price: {best_match['price']}")
        print(f"Store: {best_match['store_name']}")
        


if __name__ == "__main__":
    # Manually enter details
    product_name = "PEDRO XIMENEZ"
    producer = "Callington Mill Distillery"
    varietal = "Whiskey"
    vintage = None
    
    # Search for the product
    results = search_product(product_name, producer, varietal, vintage)
    print("All search results:")
    print("=" * 80)
    pprint(results)
    print("\n")
    
    # Filter results with fuzzy matching
    filtered_results = filter_with_fuzzy_matching(results, product_name, threshold=80)
    print(f"Filtered results (similarity >= 80):")
    print("=" * 80)
    
    for result in filtered_results:
        print(f"Score: {result['similarity_score']}")
        print(f"Product: {result['product_name']}")
        print(f"Price: {result['price']}")
        print(f"Store: {result['store_name']}")
        print(f"Rating: {result['rating']}")
        print("-" * 80)
    
    # Summarize cheapest and highest rated
    summarize_results(filtered_results)
