import os
import json
from serpapi.google_search import GoogleSearch
from dotenv import load_dotenv
from pprint import pprint
from openai import OpenAI
from fuzzywuzzy import fuzz, process
import re

load_dotenv()

# --------------------------
# API KEYS
# --------------------------
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# --------------------------
# Enhanced Fuzzy Verification with Brand Priority
# --------------------------
def verify_with_brand_priority(target_product, candidate_product, brand=None, threshold=80):
    """
    Verifies a product by prioritizing brand/producer match, then fuzzy match on descriptive text.
    
    Returns:
        is_match (bool)
        reason (str)
    """
    def clean(text):
        if not text:
            return ""
        text = re.sub(r'\s+', ' ', str(text).lower().strip())
        return text
    
    target_clean = clean(target_product)
    candidate_clean = clean(candidate_product)
    
    # Step 1: Check brand/producer exact match
    if brand:
        brand_clean = clean(brand)
        if brand_clean not in candidate_clean:
            return False, f"Brand mismatch (target: {brand_clean})"
    
    # Step 2: Fuzzy match remaining descriptive words
    ratio = fuzz.ratio(target_clean, candidate_clean)
    partial_ratio = fuzz.partial_ratio(target_clean, candidate_clean)
    token_sort_ratio = fuzz.token_sort_ratio(target_clean, candidate_clean)
    best_score = max(ratio, partial_ratio, token_sort_ratio)
    
    is_match = best_score >= threshold
    reason = f"Fuzzy match score: {best_score} (threshold: {threshold})"
    
    return is_match, reason

# --------------------------
# Safe JSON extraction
# --------------------------
def safe_json_extract(text):
    """
    Extracts JSON from LLM output reliably.
    Returns an empty list if parsing fails.
    """
    try:
        return json.loads(text)
    except:
        # Attempt to extract JSON array using regex
        match = re.search(r"\[.*\]", text, re.S)
        if match:
            try:
                return json.loads(match.group())
            except:
                return []
    return []

# --------------------------
# Google Shopping Search
# --------------------------
def search_product(product_name, producer=None, varietal=None, vintage=None, location="London"):
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
# SerpAPI Google Lens (Image Search)
# --------------------------
def search_with_google_lens(image_url):
    params = {
        "engine": "google_lens",
        "url": image_url,
        "hl": "en",
        "country": "gb",
        "type": "products",
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
# LLM Fallback Search
# --------------------------
def llm_fallback_search(product_name):
    prompt = f"""
    You are a beverage and drinks expert.
    Suggest up to 3 UK-based online or offline retail stores where someone could likely buy "{product_name}".
    Always return at least one suggestion.
    Return ONLY a JSON array with fields:
        store_name (string),
        url (string, if possible),
        reason (string explaining why the store is relevant)
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6
    )
    try:
        text_output = response.choices[0].message.content
        results = safe_json_extract(text_output)
        if not results:
            results = [{
                "store_name": "Generic UK Beverage Retailer",
                "url": "",
                "reason": f"Suggested fallback store for {product_name}"
            }]
        return results
    except:
        return [{
            "store_name": "Generic UK Beverage Retailer",
            "url": "",
            "reason": f"Suggested fallback store for {product_name}"
        }]

# --------------------------
# Hybrid Search + Verify + Lens + Fallback
# --------------------------
def hybrid_search_and_verify(product_name, image_url=None, producer=None, varietal=None, vintage=None, fuzzy_threshold=80):
    target_product = f"{product_name} {producer or ''} {varietal or ''} {vintage or ''}".strip()
    verified_results = []
    
    # Step 1: Google Shopping Search
    print(f"\n🔎 Step 1: Searching Google Shopping for: {target_product}")
    shopping_results = search_product(product_name, producer, varietal, vintage)
    
    print("All Shopping results:")
    print("=" * 80)
    pprint(shopping_results)
    print("\n")

    # Verify Shopping Results with Brand-Priority Fuzzy Matching
    print(f"✅ Verifying Google Shopping results with brand-prioritized fuzzy matching (threshold: {fuzzy_threshold})...")
    for result in shopping_results:
        match, reason = verify_with_brand_priority(target_product, result["product_name"], brand=producer, threshold=fuzzy_threshold)
        if match:
            result["verification_reason"] = reason
            result["source"] = "Google Shopping"
            verified_results.append(result)
            print(f"   ✓ Match found: {result['product_name'][:50]}...")
    
    if verified_results:
        print(f"🎯 Found {len(verified_results)} verified matches from Google Shopping")
        return verified_results
    
    # Step 2: Google Lens Search (if image available)
    if image_url:
        print(f"\n🖼️ Step 2: No verified matches from Shopping. Running Google Lens search...")
        lens_results = search_with_google_lens(image_url)
        
        print("All Google Lens results:")
        print("=" * 80)
        pprint(lens_results)
        print("\n")

        # Verify Lens Results with Brand-Priority Fuzzy Matching
        print(f"✅ Verifying Google Lens results with brand-prioritized fuzzy matching (threshold: {fuzzy_threshold})...")
        for result in lens_results:
            match, reason = verify_with_brand_priority(target_product, result["product_name"], brand=producer, threshold=fuzzy_threshold)
            if match:
                result["verification_reason"] = reason
                result["source"] = "Google Lens"
                verified_results.append(result)
                print(f"   ✓ Match found: {result['product_name'][:50]}...")
        
        if verified_results:
            print(f"🎯 Found {len(verified_results)} verified matches from Google Lens")
            return verified_results
    
    # Step 3: LLM Fallback
    print(f"\n⚠️ Step 3: No verified matches found. Using LLM fallback...")
    fallback_results = llm_fallback_search(product_name)
    
    # Format fallback results to match the structure
    formatted_fallback = []
    for idx, result in enumerate(fallback_results):
        formatted_result = {
            "product_name": product_name,
            "store_name": result.get("store_name", f"Store {idx+1}"),
            "url": result.get("url", ""),
            "reason": result.get("reason", "LLM expert suggestion"),
            "source": "LLM Fallback"
        }
        formatted_fallback.append(formatted_result)
    
    print(f"💡 LLM suggested {len(formatted_fallback)} places")
    return formatted_fallback

# --------------------------
# Summarize Results
# --------------------------
def summarize_results(filtered_results):
    if not filtered_results:
        print("\n❌ No verified matches found.")
        return
    
    print(f"\n✅ Final Results (from {filtered_results[0].get('source', 'unknown source')}):")
    print("=" * 80)
    
    for idx, place in enumerate(filtered_results, 1):
        print(f"\nPlace {idx}:")
        print(f"  Product: {place.get('product_name')}")
        print(f"  Store: {place.get('store_name')}")
        
        if place.get('price'):
            print(f"  Price: {place.get('price')}")
        if place.get('rating'):
            print(f"  Rating: {place.get('rating')}")
        if place.get('link'):
            print(f"  Link: {place.get('link')}")
        if place.get('url'):
            print(f"  Store URL: {place.get('url')}")
        if place.get('verification_reason'):
            print(f"  Verification: {place['verification_reason']}")
        if place.get('reason'):
            print(f"  Suggestion: {place['reason']}")
        if place.get('source'):
            print(f"  Source: {place['source']}")

# --------------------------
# Manual Test Runner
# --------------------------
if __name__ == "__main__":
    product_name = input("Enter product name: ").strip()
    producer = input("Enter producer (optional): ").strip() or None
    varietal = input("Enter varietal (optional): ").strip() or None
    vintage = input("Enter vintage (optional): ").strip() or None
    image_url = input("Enter image URL (optional): ").strip() or None
    threshold = input("Enter fuzzy match threshold (optional, default 80): ").strip()
    
    try:
        fuzzy_threshold = int(threshold) if threshold else 80
    except ValueError:
        fuzzy_threshold = 80

    print(f"\n🚀 Running hybrid search for: {product_name}")
    print(f"📊 Using fuzzy match threshold: {fuzzy_threshold}")

    verified_results = hybrid_search_and_verify(
        product_name, image_url, producer, varietal, vintage, fuzzy_threshold
    )

    print("\n📊 Final Results:")
    print("=" * 80)
    pprint(verified_results[:5])  # Show up to 5 results
    summarize_results(verified_results[:3])  # Summarize top 3
