import os
import json
import re
from pprint import pprint
from serpapi.google_search import GoogleSearch
from dotenv import load_dotenv
from openai import OpenAI
from fuzzywuzzy import fuzz

# --------------------------
# Load environment variables
# --------------------------
load_dotenv()
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# --------------------------
# Utility functions
# --------------------------
def clean(text):
    """Normalize text for fuzzy matching."""
    if not text:
        return ""
    return re.sub(r'\s+', ' ', str(text).lower().strip())

def safe_json_extract(text):
    """Extract JSON from text safely."""
    try:
        return json.loads(text)
    except:
        match = re.search(r"\[.*\]", text, re.S)
        if match:
            try:
                return json.loads(match.group())
            except:
                return []
    return []

# --------------------------
# Product Verification Function
# --------------------------
def verify_product(target_product, candidate_product, store_name=None, brand=None, 
                   threshold=80, high_name_threshold=90, e_commerce_sites=None):
    """
    Verify a product match using:
        1. Product name fuzzy match
        2. Brand match (if provided)
        3. E-commerce inclusion for missing brand
        4. Strong name match allows Likely even for non-e-commerce
    Returns:
        is_match (bool)
        category (str): "Verified", "Likely", "Rejected"
        reason (str)
    """
    if e_commerce_sites is None:
        e_commerce_sites = ["amazon", "ebay", "doordash", "ubereats", "tesco", "ocado"]

    target_clean = clean(target_product)
    candidate_clean = clean(candidate_product)
    store_clean = clean(store_name)

    # Step 1: Name fuzzy match
    ratio = fuzz.ratio(target_clean, candidate_clean)
    partial_ratio = fuzz.partial_ratio(target_clean, candidate_clean)
    token_sort_ratio = fuzz.token_sort_ratio(target_clean, candidate_clean)
    name_score = max(ratio, partial_ratio, token_sort_ratio)

    if name_score < threshold:
        return False, "Rejected", f"Product name fuzzy score too low: {name_score}"

    # Step 2: Brand check
    if brand:
        brand_clean = clean(brand)
        brand_score = fuzz.partial_ratio(brand_clean, candidate_clean)
        if brand_score >= 90:
            return True, "Verified", f"Name score: {name_score}, Brand matched: {brand_score}"
        else:
            if any(ecom in store_clean for ecom in e_commerce_sites):
                return True, "Likely", f"Name score: {name_score}, Brand weak but store is e-commerce"
            elif name_score >= high_name_threshold:
                return True, "Likely", f"Name score: {name_score}, Brand weak, Non-e-commerce but strong name"
            else:
                return False, "Rejected", f"Name score: {name_score}, Brand mismatch, Non-e-commerce, weak name"

    # Step 3: Brand missing
    if any(ecom in store_clean for ecom in e_commerce_sites):
        return True, "Likely", f"Name score: {name_score}, Brand missing but store is e-commerce"
    elif name_score >= high_name_threshold:
        return True, "Likely", f"Name score: {name_score}, Brand missing, Non-e-commerce but strong name"
    else:
        return False, "Rejected", f"Name score: {name_score}, Brand missing, Non-e-commerce, weak name"

# --------------------------
# Google Shopping Search
# --------------------------
def search_product(product_name, producer=None, varietal=None, vintage=None, location="London"):
    """Fetch product search results from Google Shopping via SerpAPI."""
    query_parts = [product_name]
    if producer: query_parts.append(producer)
    if varietal: query_parts.append(varietal)
    if vintage: query_parts.append(str(vintage))
    query = " ".join(query_parts)

    params = {
        "engine": "google_shopping",
        "q": query,
        "hl": "en",
        "gl": "uk",
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
            "link": item.get("link"),
            "thumbnail": item.get("thumbnail")
        }
        extracted_results.append(product_info)
    return extracted_results

# --------------------------
# Google Lens (Image) Search
# --------------------------
def search_with_google_lens(image_url):
    """Search visually using Google Lens via SerpAPI."""
    params = {
        "engine": "google_lens",
        "url": image_url,
        "hl": "en",
        "country": "gb",
        "type": "exact_matches",
        "api_key": SERPAPI_KEY,
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    extracted_results = []
    for item in results.get("visual_matches", []):
        product_info = {
            "product_name": item.get("title"),
            "store_name": item.get("source"),
            "link": item.get("link"),
            "thumbnail": item.get("thumbnail")
        }
        extracted_results.append(product_info)
    return extracted_results

# --------------------------
# Hybrid Search + Verification
# --------------------------
def hybrid_search(product_name, image_url=None, producer=None, varietal=None, vintage=None,
                  fuzzy_threshold=80, high_name_threshold=90):
    """
    Full hybrid search workflow with debug prints:
        1. Google Shopping
        2. Google Lens (optional)
        3. Verification with name + brand + e-commerce logic
        4. Limit final output to top 3 (with at least one Likely)
    """
    target_product = f"{product_name} {producer or ''} {varietal or ''} {vintage or ''}".strip()
    verified_results = []

    # Step 1: Google Shopping
    print("\n🔎 Searching Google Shopping...")
    shopping_results = search_product(product_name, producer, varietal, vintage)
    if not shopping_results:
        print("⚠️ No shopping results found.")
    else:
        print(f"Found {len(shopping_results)} shopping results.")
        print(shopping_results)
    
    for result in shopping_results:
        match, category, reason = verify_product(
            target_product, result["product_name"], result.get("store_name"), brand=producer,
            threshold=fuzzy_threshold, high_name_threshold=high_name_threshold
        )
        if match:
            result.update({
                "verification_reason": reason,
                "source": "Google Shopping",
                "category": category
            })
            verified_results.append(result)

    if not verified_results:
        print("❌ No verified results in shopping, trying Lens..." if image_url else "❌ No verified results in shopping.")

    # Step 2: Google Lens (if image provided)
    if image_url and (not shopping_results or not verified_results):
        print("\n📸 Searching Google Lens with provided image...")
        lens_results = search_with_google_lens(image_url)
        if not lens_results:
            print("⚠️ No lens results found.")
        else:
            print(f"Found {len(lens_results)} Lens results.")
        for result in lens_results:
            match, category, reason = verify_product(
                target_product, result["product_name"], result.get("store_name"), brand=producer,
                threshold=fuzzy_threshold, high_name_threshold=high_name_threshold
            )
            if match:
                result.update({
                    "verification_reason": reason,
                    "source": "Google Lens",
                    "category": category
                })
                verified_results.append(result)
        if not lens_results:
            print("❌ No matches found via Lens.")

    # --------------------------
    # Step 3: LLM Fallback (commented out for now)
    # --------------------------
    # fallback_results = llm_fallback_search(product_name)
    # for idx, result in enumerate(fallback_results):
    #     formatted_result = {
    #         "product_name": product_name,
    #         "store_name": result.get("store_name", f"Store {idx+1}"),
    #         "url": result.get("url", ""),
    #         "reason": result.get("reason", "LLM expert suggestion"),
    #         "source": "LLM Fallback",
    #         "category": "Likely"
    #     }
    #     verified_results.append(formatted_result)
    # --------------------------

    # Step 4: Ensure at least one Likely result is included
    final_results = []
    likely_added = False
    for result in verified_results:
        if result["category"] == "Likely" and not likely_added:
            final_results.append(result)
            likely_added = True
    for result in verified_results:
        if result not in final_results and len(final_results) < 3:
            final_results.append(result)

    print(f"\n✅ Total verified/likely results to show: {len(final_results)}")
    return final_results[:3]  # Return only top 3 places

# --------------------------
# Display/Summarize Results
# --------------------------
def summarize_results(results):
    """Print top results in a user-friendly format."""
    if not results:
        print("\n❌ No verified matches found.")
        return

    print(f"\n✅ Top {len(results)} Places Where You Can Get The Product:")
    print("=" * 80)
    for idx, place in enumerate(results, 1):
        print(f"\nPlace {idx}:")
        print(f"  Product: {place.get('product_name')}")
        print(f"  Store: {place.get('store_name')}")
        if place.get("price"):
            print(f"  Price: {place.get('price')}")
        if place.get("rating"):
            print(f"  Rating: {place.get('rating')}")
        if place.get("link"):
            print(f"  Link: {place.get('link')}")
        if place.get("verification_reason"):
            print(f"  Verification: {place['verification_reason']}")
        if place.get("category"):
            print(f"  Category: {place['category']}")
        if place.get("source"):
            print(f"  Source: {place['source']}")

# --------------------------
# Main Runner
# --------------------------
if __name__ == "__main__":
    product_name = input("Enter product name: ").strip()
    producer = input("Enter producer (optional): ").strip() or None
    varietal = input("Enter varietal (optional): ").strip() or None
    vintage = input("Enter vintage (optional): ").strip() or None
    image_url = input("Enter image URL (optional): ").strip() or None
    
    fuzzy_threshold = 85

    print(f"\n🚀 Running hybrid search for: {product_name} with threshold {fuzzy_threshold}")
    verified_results = hybrid_search(
        product_name, image_url, producer, varietal, vintage, fuzzy_threshold
    )

    pprint(verified_results)  # Show raw top 3 results
    summarize_results(verified_results)
