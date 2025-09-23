import os
import json
import numpy as np
from serpapi.google_search import GoogleSearch
from dotenv import load_dotenv
from pprint import pprint
from openai import OpenAI

# Load API keys
load_dotenv()
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

# ===================== SerpAPI Search =====================

def search_product(product_name, producer=None, varietal=None, vintage=None, location="London"):
    """
    Search for a product on Google Shopping using SerpAPI
    and return simplified results.
    """
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

    extracted_results = []
    for item in results.get("shopping_results", []):
        product_info = {
            "product_name": item.get("title"),
            "store_name": item.get("source"),
            "price": item.get("price"),
            "rating": item.get("rating"),
            "link": item.get("link")
        }
        extracted_results.append(product_info)

    return extracted_results

# ===================== Embeddings =====================

def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def get_embeddings_batch(texts, model="text-embedding-3-small"):
    """Get embeddings for a list of texts in one API call."""
    response = client.embeddings.create(
        model=model,
        input=texts
    )
    return [np.array(item.embedding) for item in response.data]

def filter_with_embeddings(results, target_product, threshold=0.75):
    """
    Filter results using embeddings and negative keyword exclusion.
    """
    negative_keywords = [
        "empty", "bottle only", "decanted", "decant", "used", 
        "vintage bottle", "old bottle", "collector", "display",
        "ornamental", "set of", "pack of", "lot of", "souvenir"
    ]

    # Prepare texts for embedding: target + candidate names
    product_names = [item["product_name"].lower() for item in results]
    texts_to_embed = [target_product] + product_names

    # Get all embeddings in one request
    embeddings = get_embeddings_batch(texts_to_embed)
    target_embedding = embeddings[0]
    product_embeddings = embeddings[1:]

    matches = []
    for item, embedding in zip(results, product_embeddings):
        product_name = item["product_name"].lower()

        # Skip if contains negative keywords
        if any(keyword in product_name for keyword in negative_keywords):
            continue

        # Compute similarity
        score = cosine_similarity(target_embedding, embedding)

        if score >= threshold:
            item_with_score = item.copy()
            item_with_score["similarity_score"] = score
            matches.append(item_with_score)

    # Sort by similarity score
    matches.sort(key=lambda x: x["similarity_score"], reverse=True)

    return matches

# ===================== Main =====================

if __name__ == "__main__":
    product_name = "W. L. WELLER ANTIQUE 107"
    producer = "Sazerac"
    varietal = None
    vintage = None

    results = search_product(product_name, producer, varietal, vintage)
    
    print("All search results:")
    print("=" * 80)
    pprint(results)
    print("\n")

    filtered_results = filter_with_embeddings(
        results,
        product_name,
        threshold=0.75
    )

    print(f"Filtered results (similarity >= 0.75):")
    print("=" * 80)
    
    for result in filtered_results:
        print(f"Score: {result['similarity_score']:.3f}")
        print(f"Product: {result['product_name']}")
        print(f"Price: {result['price']}")
        print(f"Store: {result['store_name']}")
        print(f"Rating: {result['rating']}")
        print("-" * 80)

    if filtered_results:
        best_match = filtered_results[0]
        print("Best match:")
        print(f"Product: {best_match['product_name']}")
        print(f"Price: {best_match['price']}")
        print(f"Store: {best_match['store_name']}")
        print(f"Rating: {best_match['rating']}")
        print(f"Similarity Score: {best_match['similarity_score']:.3f}")
    else:
        print("No matches found with the given criteria.")
