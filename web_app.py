from flask import Flask, request, render_template, jsonify
from datetime import datetime
import json, re, os, uuid
import math
import requests
from fuzzywuzzy import process

app = Flask(__name__)
app.config["SECRET_KEY"] = "secret!"

# --------------------------
# BM25 Implementation
# --------------------------
import math
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

class BM25:
    def __init__(self, corpus, tokenizer=None, fuzzy_threshold=80):
        self.corpus_size = len(corpus)
        self.corpus = corpus
        self.avgdl = 0
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self.tokenizer = tokenizer if tokenizer else self._default_tokenizer
        
        # Parameters for BM25
        self.k1 = 1.5
        self.b = 0.75
        
        # Parameter for fuzzy matching
        self.fuzzy_threshold = fuzzy_threshold
        
        # Create term-to-document mapping for fuzzy search
        self.term_mapping = {}
        
        # Preprocess corpus
        self._initialize(corpus)
    
    def _default_tokenizer(self, text):
        # Simple tokenization by splitting on whitespace and lowercase
        return text.lower().split()
    
    def _initialize(self, corpus):
        # Count document frequencies for each term
        for doc_idx, document in enumerate(corpus):
            if document:
                tokens = self.tokenizer(document)
                self.doc_len.append(len(tokens))
                
                # Count term frequencies
                freq = {}
                for token in tokens:
                    freq[token] = freq.get(token, 0) + 1
                    
                    # Update term mapping for fuzzy search
                    if token not in self.term_mapping:
                        self.term_mapping[token] = set()
                    self.term_mapping[token].add(doc_idx)
                    
                self.doc_freqs.append(freq)
                
                # Update document frequencies
                for token in freq:
                    self.idf[token] = self.idf.get(token, 0) + 1
        
        # Calculate average document length
        self.avgdl = sum(self.doc_len) / self.corpus_size if self.corpus_size > 0 else 0
        
        # Calculate inverse document frequency
        for token, freq in self.idf.items():
            self.idf[token] = math.log(1 + (self.corpus_size - freq + 0.5) / (freq + 0.5))
    
    def _find_fuzzy_matches(self, token):
        """Find fuzzy matches for a token above the threshold"""
        if not self.term_mapping:
            return set()
            
        # Find the best fuzzy matches
        matches = process.extract(
            token, 
            self.term_mapping.keys(), 
            limit=10,  # Limit the number of fuzzy matches to consider
            scorer=fuzz.ratio  # Can be changed to partial_ratio, token_sort_ratio, etc.
        )
        
        # Filter matches above threshold and get their document indices
        doc_indices = set()
        for match, score in matches:
            if score >= self.fuzzy_threshold:
                doc_indices.update(self.term_mapping.get(match, set()))
        
        return doc_indices
    
    def get_scores(self, query):
        query_tokens = self.tokenizer(query)
        scores = [0] * self.corpus_size
        
        # Process each query token
        for token in query_tokens:
            # Get exact matches
            exact_match = False
            if token in self.idf:
                exact_match = True
                q_freq = query_tokens.count(token)
                
                for i, doc_freq in enumerate(self.doc_freqs):
                    if token not in doc_freq:
                        continue
                        
                    # Standard BM25 scoring formula
                    doc_term_freq = doc_freq[token]
                    doc_length = self.doc_len[i]
                    
                    numerator = self.idf[token] * doc_term_freq * (self.k1 + 1)
                    denominator = doc_term_freq + self.k1 * (1 - self.b + self.b * doc_length / self.avgdl)
                    
                    scores[i] += (numerator / denominator) if denominator != 0 else 0
            
            # If no exact match or we want to augment with fuzzy matches
            if not exact_match or True:  # Always try fuzzy matches to improve results
                # Get fuzzy matches above threshold
                fuzzy_doc_indices = self._find_fuzzy_matches(token)
                
                # Apply a fuzzy bonus to matching documents (with a penalty factor)
                fuzzy_bonus = 0.5  # Adjust this factor to control fuzzy match impact
                for doc_idx in fuzzy_doc_indices:
                    # Calculate similarity ratio for weighting
                    best_token_in_doc = max(
                        self.tokenizer(self.corpus[doc_idx]), 
                        key=lambda t: fuzz.ratio(t, token),
                        default=""
                    )
                    similarity = fuzz.ratio(token, best_token_in_doc) / 100.0
                    
                    # Add a weighted bonus to the score
                    if token in self.idf:
                        token_idf = self.idf[token]
                    else:
                        # Estimate IDF for terms not in corpus
                        token_idf = math.log(1 + self.corpus_size)
                        
                    scores[doc_idx] += token_idf * fuzzy_bonus * similarity
        
        return scores
    
    def get_top_n(self, query, documents=None, n=5, include_scores=False):
        """Return top N documents for a query with optional score inclusion"""
        docs = documents if documents is not None else self.corpus
        
        if len(docs) != self.corpus_size:
            raise ValueError("The number of documents doesn't match the corpus size")
            
        scores = self.get_scores(query)
        top_n = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:n]
        
        if include_scores:
            return [(docs[i], score) for i, score in top_n if score > 0]
        else:
            return [docs[i] for i, score in top_n if score > 0]
    
    def search(self, query, documents=None, n=5, include_scores=False, min_score=0):
        """User-friendly search function with minimum score threshold"""
        docs = documents if documents is not None else self.corpus
        
        if len(docs) != self.corpus_size:
            raise ValueError("The number of documents doesn't match the corpus size")
            
        scores = self.get_scores(query)
        # Filter by minimum score and sort
        results = [(i, score) for i, score in enumerate(scores) if score > min_score]
        results = sorted(results, key=lambda x: x[1], reverse=True)[:n]
        
        if include_scores:
            return [(docs[i], score) for i, score in results]
        else:
            return [docs[i] for i, score in results]



   

def load_menu(menu_file='menu.json'):
    with open(menu_file, 'r', encoding="utf-8") as file:
        data = json.load(file)
    if isinstance(data, list):
        items = []
        for category in data:
            items.extend(category.get("items", []))
        return items
    elif isinstance(data, dict):
        return data.get("items", [])
    else:
        return []

menu = load_menu()

# Initialize BM25 for English and Arabic menu search
def initialize_bm25_search():
    # Prepare English corpus
    english_corpus = [dish.get('name', '').lower() for dish in menu]
    english_bm25 = BM25(english_corpus)
    
    # Prepare Arabic corpus
    arabic_corpus = [dish.get('name_arabic', '').lower() for dish in menu]
    arabic_bm25 = BM25(arabic_corpus)
    
    return english_bm25, arabic_bm25

english_bm25, arabic_bm25 = initialize_bm25_search()

# ---------------------------
# Language Detection Helper
# ---------------------------
def detect_language(text):
    if re.search(r'[\u0600-\u06FF]', text):
        return "ar"
    return "en"

def get_user_lang(user_number):
    return user_states.get(user_number + "_lang", "en")

# ---------------------------
# Get Dish by Name (Using BM25)
# ---------------------------
def normalize_arabic(text):
    text = re.sub(r'[ًٌٍَُِّْـ]', '', text)
    text = re.sub(r'[إأآا]', 'ا', text)
    text = re.sub(r'ى', 'ي', text)
    text = re.sub(r'ؤ', 'ء', text)
    text = re.sub(r'ئ', 'ء', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip().lower()

def get_dish_by_name(query, lang="en"):
    # Normalize query
    if lang == "ar":
        norm_query = normalize_arabic(query)
        
        # First try exact match
        for dish in menu:
            dish_name = dish.get('name_arabic', '')
            if normalize_arabic(dish_name) == norm_query:
                return dish
        
        # If no exact match, use BM25
        results = arabic_bm25.get_top_n(norm_query, menu, n=1, include_scores=True)  # Add include_scores=True
        if results and len(results) > 0:
            # Check if results is non-empty and has the expected (dish, score) tuple format
            if isinstance(results[0], tuple) and len(results[0]) > 1:
                if results[0][1] > 0.5:  # Score threshold
                    return results[0][0]
            else:
                # If results[0] is not a tuple with score, just return the first result
                return results[0]
            
        # Fallback to fuzzy matching as a last resort
        names_list = [normalize_arabic(dish.get('name_arabic', dish.get('name', ''))) for dish in menu]
        best_match, score = process.extractOne(norm_query, names_list)
        if score >= 70:
            for dish in menu:
                dish_name = dish.get('name_arabic', dish.get('name', ''))
                if normalize_arabic(dish_name) == best_match:
                    return dish
    else:
        norm_query = query.strip().lower()
        
        # First try exact match
        for dish in menu:
            if dish.get('name', '').strip().lower() == norm_query:
                return dish
        
        # If no exact match, use BM25
        results = english_bm25.get_top_n(norm_query, menu, n=1, include_scores=True)  # Add include_scores=True
        if results and len(results) > 0:
            # Check if results is non-empty and has the expected (dish, score) tuple format
            if isinstance(results[0], tuple) and len(results[0]) > 1:
                if results[0][1] > 0.5:  # Score threshold
                    return results[0][0]
            else:
                # If results[0] is not a tuple with score, just return the first result
                return results[0]
            
        # Fallback to fuzzy matching as a last resort
        names_list = [dish.get('name', '').strip().lower() for dish in menu]
        best_match, score = process.extractOne(norm_query, names_list)
        if score >= 70:
            for dish in menu:
                if dish.get('name', '').strip().lower() == best_match:
                    return dish
    
    return None
# ---------------------------
# Multi-Item Order Parser - IMPROVED
# ---------------------------
def parse_multi_order(text):
    if not text:
        return []
        
    text = text.lower()
    text = re.sub(r"(i want to order|order|please|أريد|طلب)", "", text)
    
    # Enhanced pattern to better catch quantity and dish pairs
    pattern = re.compile(r'(\d+|one|two|three|four|five|six|seven|eight|nine|ten|واحد|اثنين|ثلاثة|أربعة|خمسة|ستة|سبعة|ثمانية|تسعة|عشرة)\s+([a-z\u0600-\u06FF\s]+?)(?=\s*(?:and|,|و|\d+|one|two|three|four|five|six|seven|eight|nine|ten|واحد|اثنين|ثلاثة|أربعة|خمسة|ستة|سبعة|ثمانية|تسعة|عشرة|$))')
    
    matches = pattern.findall(text)
    result = []
    
    word_to_num = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "واحد": 1, "اثنين": 2, "ثلاثة": 3, "أربعة": 4, "خمسة": 5,
        "ستة": 6, "سبعة": 7, "ثمانية": 8, "تسعة": 9, "عشرة": 10
    }
    
    for qty_str, dish in matches:
        qty = int(qty_str) if qty_str.isdigit() else word_to_num.get(qty_str, 1)
        dish_name = dish.strip()
        if dish_name:  # Only add non-empty dish names
            result.append((qty, dish_name))
    
    # If no matches found but text looks like a single dish order without explicit quantity
    if not result and text.strip():
        result.append((1, text.strip()))
        
    return result

# ---------------------------
# Main Menu Display Function (with images)
# ---------------------------
def show_menu(lang="en"):
    menu_images = [
        "https://i.ibb.co/FL6SCT67/1.jpg",
        "https://i.ibb.co/x8G4mt8C/2.jpg",
        "https://i.ibb.co/ZZLXgsz/3.jpg",
        "https://i.ibb.co/fdWDRwTq/4.jpg"
    ]
    if lang == "ar":
        return {
            "text": "🍽️ هذه هي قائمتنا! \n\nهل ترغب في تقديم طلب؟ )",
            "menu_images": menu_images
        }
    else:
        return {
            "text": "🍽️ Here's our menu! \n\nWould you like to place an order? ",
            "menu_images": menu_images
        }

def extract_quantity(text):
    qty_match = re.findall(r'\d+', text)
    if qty_match:
        return int(qty_match[0])
    word_to_num = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "واحد": 1, "اثنين": 2, "ثلاثة": 3, "أربعة": 4, "خمسة": 5,
        "ستة": 6, "سبعة": 7, "ثمانية": 8, "تسعة": 9, "عشرة": 10
    }
    for word in text.lower().split():
        if word in word_to_num:
            return word_to_num[word]
    return 1

# ---------------------------
# Show Current Order Summary
# ---------------------------
def show_order_summary(user_number, lang="en"):
    if not order_list.get(user_number):
        if lang == "ar":
            return "⚠️ لا يوجد عناصر في طلبك حالياً."
        else:
            return "⚠️ There are no items in your order currently."
    
    summary = ""
    if lang == "ar":
        summary = "🛒 طلبك الحالي:\n"
        for i, item in enumerate(order_list[user_number], 1):
            name = item.get("name_arabic", item.get("name", "طبق غير معروف"))
            qty = item.get("quantity", 1)
            customizations = item.get("customizations", "لا تخصيصات")
            summary += f"{i}. {name} (الكمية: {qty})"
            if customizations and customizations != "None":
                summary += f" - التخصيصات: {customizations}"
            summary += "\n"
    else:
        summary = "🛒 Your current order:\n"
        for i, item in enumerate(order_list[user_number], 1):
            name = item.get("name", "Unknown Dish")
            qty = item.get("quantity", 1)
            customizations = item.get("customizations", "No customizations")
            summary += f"{i}. {name} (Qty: {qty})"
            if customizations and customizations != "None":
                summary += f" - Customizations: {customizations}"
            summary += "\n"
    
    return summary

# ---------------------------
# Groq API Integration
# ---------------------------
def enhance_response_with_groq(response_message, user_input, lang="en"):
    """
    Enhance the bot's response using Groq LLaMA model
    """
    # Groq API configuration
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "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")
    GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "llama3-70b-8192",
        "messages": [
            {
                "role": "user",
                "content": f"Task: Rewrite the following message to be brief, conversational, and human-like, while keeping *every* original detail exactly the same. Message to rewrite: '{response_message}'. if the message to rewrite isnt a greetings dont greet the user .Output *only* the rewritten message. If the message is empty, state 'Error: Message is empty.'"
            }        
        ],
        "temperature": 0.7,
        "max_tokens": 1024
    }
    
    try:
        # Call Groq API
        response = requests.post(GROQ_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        
        # Extract enhanced response
        enhanced_response = response.json()["choices"][0]["message"]["content"]
        if enhanced_response.startswith(("\"", "'")) and enhanced_response.endswith(("\"", "'")):
            enhanced_response = enhanced_response[1:-1].strip()

        return enhanced_response
    except Exception as e:
        print(f"Error calling Groq API: {e}")
        # Fall back to original response if API call fails
        return response_message

# ---------------------------
# In-memory State Management
# ---------------------------
user_states = {}   # Tracks current state per user.
order_list = {}    # Stores ordered dishes per user.
temp_order = {}    # Temporarily holds dish info before customization.

def finalize_order_to_json(final_order):
    try:
        with open('orders_history.json', 'r', encoding='utf-8') as f:
            history = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        history = []
    history.append(final_order)
    with open('orders_history.json', 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=4, ensure_ascii=False)
    
    # Make sure orders directory exists
    os.makedirs('orders', exist_ok=True)
    
    filename = f"orders/order_{final_order['order_id']}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(final_order, f, indent=4, ensure_ascii=False)

def reset_user(user_number):
    user_states[user_number] = "awaiting_menu_or_order"
    order_list[user_number] = []
    keys_to_remove = [key for key in list(user_states.keys()) if key.startswith(user_number + "_")]
    for key in keys_to_remove:
        del user_states[key]

def reset_server():
    global user_states, order_list, temp_order
    user_states.clear()
    order_list.clear()
    temp_order.clear()

# ---------------------------
# Order Modification Functions
# ---------------------------
def remove_item(user_number, item_index, lang="en"):
    if not order_list.get(user_number) or item_index <= 0 or item_index > len(order_list[user_number]):
        if lang == "ar":
            return "❌ رقم العنصر غير صالح. الرجاء المحاولة مرة أخرى."
        else:
            return "❌ Invalid item number. Please try again."
    
    removed_item = order_list[user_number].pop(item_index - 1)
    
    if lang == "ar":
        name = removed_item.get("name_arabic", removed_item.get("name", "طبق غير معروف"))
        return f"✅ تمت إزالة {name} من طلبك.\n\n{show_order_summary(user_number, lang)}"
    else:
        name = removed_item.get("name", "Unknown Dish")
        return f"✅ Removed {name} from your order.\n\n{show_order_summary(user_number, lang)}"

def modify_item_quantity(user_number, item_index, new_quantity, lang="en"):
    if not order_list.get(user_number) or item_index <= 0 or item_index > len(order_list[user_number]):
        if lang == "ar":
            return "❌ رقم العنصر غير صالح. الرجاء المحاولة مرة أخرى."
        else:
            return "❌ Invalid item number. Please try again."
    
    item = order_list[user_number][item_index - 1]
    old_quantity = item.get("quantity", 1)
    item["quantity"] = new_quantity
    
    if lang == "ar":
        name = item.get("name_arabic", item.get("name", "طبق غير معروف"))
        return f"✅ تم تغيير كمية {name} من {old_quantity} إلى {new_quantity}.\n\n{show_order_summary(user_number, lang)}"
    else:
        name = item.get("name", "Unknown Dish")
        return f"✅ Changed quantity of {name} from {old_quantity} to {new_quantity}.\n\n{show_order_summary(user_number, lang)}"

def modify_item_customization(user_number, item_index, new_customization, lang="en"):
    if not order_list.get(user_number) or item_index <= 0 or item_index > len(order_list[user_number]):
        if lang == "ar":
            return "❌ رقم العنصر غير صالح. الرجاء المحاولة مرة أخرى."
        else:
            return "❌ Invalid item number. Please try again."
    
    item = order_list[user_number][item_index - 1]
    item["customizations"] = new_customization
    
    if lang == "ar":
        name = item.get("name_arabic", item.get("name", "طبق غير معروف"))
        return f"✅ تم تحديث تخصيصات {name}.\n\n{show_order_summary(user_number, lang)}"
    else:
        name = item.get("name", "Unknown Dish")
        return f"✅ Updated customizations for {name}.\n\n{show_order_summary(user_number, lang)}"

# ---------------------------
# Main Message Processor (ENHANCED)
# ---------------------------
def process_message(user_number, user_input, latitude=None, longitude=None):
    # Handle greeting or first-time interaction
    user_input = user_input.strip().lower() if user_input else ""
    greeting_keywords = {"hi", "hello", "hey", "good morning", "good afternoon", "good evening", "مرحبا", "أهلا", "السلام عليكم"}
    
    # Cancel order commands
    cancel_commands = {"cancel", "reset", "start over", "إلغاء", "إعادة تعيين", "البدء من جديد"}
    if any(cmd in user_input for cmd in cancel_commands):
        lang = get_user_lang(user_number) if user_number in user_states else detect_language(user_input)
        # If user is in the middle of an order, ask for confirmation
        if user_number in user_states and order_list.get(user_number):
            user_states[user_number] = "confirm_cancel"
            if lang == "ar":
                return enhance_response_with_groq("⚠️ هل أنت متأكد من أنك تريد إلغاء طلبك الحالي؟ (نعم/لا)", user_input, lang)
            else:
                return enhance_response_with_groq("⚠️ Are you sure you want to cancel your current order? ", user_input, lang)
        # Otherwise just reset
        reset_user(user_number)
        if lang == "ar":
            return enhance_response_with_groq("✅ تم إعادة تعيين المحادثة. يمكنك بدء طلب جديد.", user_input, lang)
        else:
            return enhance_response_with_groq("✅ Conversation reset. You can start a new order.", user_input, lang)
    
    # Handle cancellation confirmation
    if user_number in user_states and user_states[user_number] == "confirm_cancel":
        lang = get_user_lang(user_number)
        if any(word in user_input for word in ["yes", "yeah", "yep", "نعم", "أجل"]):
            reset_user(user_number)
            if lang == "ar":
                return enhance_response_with_groq("✅ تم إلغاء طلبك. يمكنك بدء طلب جديد متى أردت.", user_input, lang)
            else:
                return enhance_response_with_groq("✅ Your order has been canceled. You can start a new order whenever you're ready.", user_input, lang)
        else:
            # If not confirmed, return to previous state
            user_states[user_number] = "awaiting_menu_or_order" if "awaiting_menu_or_order" not in user_states.get(user_number, "") else user_states[user_number]
            if lang == "ar":
                return enhance_response_with_groq("👍 تم الاستمرار بطلبك الحالي.", user_input, lang)
            else:
                return enhance_response_with_groq("👍 Continuing with your current order.", user_input, lang)
    
    # Show order commands
    show_order_commands = {"show order", "view order", "my order", "current order", "عرض الطلب", "طلبي", "الطلب الحالي"}
    if any(cmd in user_input for cmd in show_order_commands):
        if user_number in user_states:
            lang = get_user_lang(user_number)
            summary = show_order_summary(user_number, lang)
            
            if order_list.get(user_number):
                if lang == "ar":
                    additional_options = ("📝 خيارات التعديل:\n"
                                         "- 'إزالة الطبق 1' لإزالة طبق\n"
                                         "- 'تغيير كمية الطبق 2 إلى 3' لتغيير الكمية\n"
                                         "- 'تعديل تخصيصات الطبق 1 إلى بدون بصل' لتغيير التخصيصات\n"
                                         "- 'متابعة الطلب' للمتابعة\n"
                                         "- 'إلغاء' لإلغاء الطلب")
                else:
                    additional_options = ("📝 Modification options:\n"
                                         "- 'Remove dish 1' to remove a dish\n"
                                         "- 'Change quantity of dish 2 to 3' to change quantity\n"
                                         "- 'Modify customizations of dish 1 to no onions' to change customizations\n"
                                         "- 'Continue order' to proceed\n"
                                         "- 'Cancel' to cancel the order")
                
                user_states[user_number] = "modifying_order"
                return enhance_response_with_groq(f"{summary}\n\n{additional_options}", user_input, lang)
            
            return enhance_response_with_groq(summary, user_input, lang)
    
    # Modification commands processing
    if user_number in user_states and user_states[user_number] == "modifying_order":
        lang = get_user_lang(user_number)
        
        # Check for continue order command
        continue_commands = {"continue", "proceed", "done", "finish", "متابعة", "استمرار", "تم", "إنتهى"}
        if any(cmd in user_input for cmd in continue_commands):
            if not order_list.get(user_number):
                if lang == "ar":
                    return enhance_response_with_groq("⚠️ لا يوجد عناصر في طلبك حالياً. الرجاء إضافة طبق أولاً.", user_input, lang)
                else:
                    return enhance_response_with_groq("⚠️ There are no items in your order currently. Please add a dish first.", user_input, lang)
            
            user_states[user_number] = "awaiting_order_type"
            if lang == "ar":
                return enhance_response_with_groq("✅ جميع الأطباق جاهزة. هل طلبك للتوصيل أم ستأتي إلى المطعم؟", user_input, lang)
            else:
                return enhance_response_with_groq("✅ All dishes are ready. Is your order for delivery or will you come to the restaurant?", user_input, lang)
        
        # Check for remove dish command
        remove_match = re.search(r'remove dish (\d+)|إزالة الطبق (\d+)', user_input, re.IGNORECASE)
        if remove_match:
            item_index = int(remove_match.group(1) if remove_match.group(1) else remove_match.group(2))
            return enhance_response_with_groq(remove_item(user_number, item_index, lang), user_input, lang)
        
        # Check for change quantity command
        qty_match = re.search(r'change quantity of dish (\d+) to (\d+)|تغيير كمية الطبق (\d+) إلى (\d+)', user_input, re.IGNORECASE)
        if qty_match:
            if qty_match.group(1):  # English match
                item_index, new_qty = int(qty_match.group(1)), int(qty_match.group(2))
            else:  # Arabic match
                item_index, new_qty = int(qty_match.group(3)), int(qty_match.group(4))
            return enhance_response_with_groq(modify_item_quantity(user_number, item_index, new_qty, lang), user_input, lang)
        
        # Check for modify customizations command
        custom_match = re.search(r'modify customizations? of dish (\d+) to (.+)|تعديل تخصيصات الطبق (\d+) إلى (.+)', user_input, re.IGNORECASE)
        if custom_match:
            if custom_match.group(1):  # English match
                item_index, new_custom = int(custom_match.group(1)), custom_match.group(2)
            else:  # Arabic match
                item_index, new_custom = int(custom_match.group(3)), custom_match.group(4)
            return enhance_response_with_groq(modify_item_customization(user_number, item_index, new_custom.strip(), lang), user_input, lang)
        
        # Default response if no modification command is detected
        if lang == "ar":
            return enhance_response_with_groq("❓ لم أفهم طلبك. الرجاء اختيار أحد خيارات التعديل أو كتابة 'متابعة الطلب' للمتابعة.", user_input, lang)
        else:
            return enhance_response_with_groq("❓ I didn't understand your request. Please choose one of the modification options or type 'continue order' to proceed.", user_input, lang)
    
    if user_input and user_input.split() and user_input.split()[0] in greeting_keywords:
        reset_user(user_number)
        lang = detect_language(user_input)
        user_states[user_number] = "awaiting_menu_or_order"
        order_list[user_number] = []
        user_states[user_number + "_lang"] = lang
        
        response = ""
        if lang == "ar":
            response = "👋 مرحباً بكم في مطعم الفرّوج! \nهل ترغب في رؤية القائمة أو تقديم طلب؟ حاول أن تقول، 'أريد رؤية القائمة' أو 'أريد طلب طعام'."
        else:
            response = "👋 Welcome to Al Farooj Restaurant! \nWould you like to see our menu or place an order? "
        
        # Enhance with Groq
        return enhance_response_with_groq(response, user_input, lang)
    
    if user_number not in user_states:
        lang = detect_language(user_input)
        user_states[user_number] = "awaiting_menu_or_order"
        order_list[user_number] = []
        user_states[user_number + "_lang"] = lang
        
        response = ""
        if lang == "ar":
            response = "👋 مرحباً بكم في مطعم الفرّوج! \nهل ترغب في رؤية القائمة أو تقديم طلب؟ حاول أن تقول، 'أريد رؤية القائمة' أو 'أريد طلب طعام'."
        else:
            response = "👋 Welcome to Al Farooj Restaurant! \nWould you like to see our menu or place an order? ."
        
        # Enhance with Groq
        return enhance_response_with_groq(response, user_input, lang)
    
    lang = get_user_lang(user_number)
    
    # --- 1. Awaiting Menu or Order Choice ---
    if user_states[user_number] == "awaiting_menu_or_order":
        if "menu" in user_input or "قائمة" in user_input:
            # Menu display doesn't need Groq enhancement
            return show_menu(lang)
        elif "order" in user_input or "طلب" in user_input or parse_multi_order(user_input):
            multi = parse_multi_order(user_input)
            if multi and len(multi) > 0:
                # Store all parsed items in a queue for processing
                user_states[user_number + "_multi"] = multi
                user_states[user_number + "_multi_index"] = 0
                user_states[user_number] = "awaiting_dish_selection"
                
                # Process the first item immediately
                qty, dish_str = multi[0]
                user_states[user_number + "_multi_index"] = 1
                
                dish = get_dish_by_name(dish_str, lang)
                if dish:
                    user_states[user_number + "_quantity"] = qty
                    temp_order[user_number] = {"dish": dish, "quantity": qty, "customizations": None}
                    user_states[user_number] = "customizing_dish"
                    
                    response = ""
                    if lang == "ar":
                        dish_name = dish.get("name_arabic", dish.get("name", ""))
                        description = dish.get("description_arabic", dish.get("description", "لا يوجد وصف."))
                        response = (f"🍽️ لقد اخترت {qty} × {dish_name}.\n"
                                f"الوصف: {description}\n"
                                "هل ترغب في تخصيصه؟ (أجب بـ 'نعم' أو 'لا')")
                    else:
                        dish_name = dish.get("name", "")
                        description = dish.get("description", "No description available.")
                        response = (f"🍽️ You selected {qty} x {dish_name}.\n"
                                f"Description: {description}\n"
                                "Would you like to customize it? (Reply with 'yes' or 'no')")
                    
                    # Enhance with Groq
                    return enhance_response_with_groq(response, user_input, lang)
                else:
                    # If dish not found, ask for clarification
                    if lang == "ar":
                        response = f"❌ لم يتم العثور على الطبق: {dish_str}. الرجاء المحاولة مرة أخرى."
                    else:
                        response = f"❌ Could not find dish: {dish_str}. Please try again."
                    
                    user_states[user_number] = "awaiting_dish_selection"
                    return enhance_response_with_groq(response, user_input, lang)
            else:
                user_states[user_number] = "awaiting_dish_selection"
                if lang == "ar":
                    response = "🍴 ماذا ترغب في طلبه؟ يمكنك كتابة اسم الطبق والكمية، مثل '2 برجر دجاج' أو '1 شاورما'."
                else:
                    response = "🍴 What would you like to order? You can type the dish name and quantity, like '2 chicken burgers' or '1 shawarma'."
                return enhance_response_with_groq(response, user_input, lang)
        else:
            response = ""
            if lang == "ar":
                response = "❓ الرجاء الرد بـ 'قائمة' أو 'طلب' للمتابعة."
            else:
                response = "❓ Please reply with 'menu' or 'order' to proceed."
            
            # Enhance with Groq
            return enhance_response_with_groq(response, user_input, lang)
    
    # --- 2. Awaiting Dish Selection ---
    elif user_states[user_number] == "awaiting_dish_selection":
        # Process multi-item orders that are already in progress
        if user_states.get(user_number + "_multi") and user_states.get(user_number + "_multi_index", 0) < len(user_states[user_number + "_multi"]):
            idx = user_states[user_number + "_multi_index"]
            qty, dish_str = user_states[user_number + "_multi"][idx]
            user_states[user_number + "_multi_index"] = idx + 1
            
            dish = get_dish_by_name(dish_str, lang)
            if not dish:
                response = ""
                if lang == "ar":
                    response = f"❌ لم يتم العثور على الطبق: {dish_str}. سنتخطى هذا العنصر."
                else:
                    response = f"❌ Could not match dish: {dish_str}. We'll skip this item."
                
                # Check if there are more items to process
                if user_states[user_number + "_multi_index"] < len(user_states[user_number + "_multi"]):
                    return process_message(user_number, "", latitude, longitude)
                else:
                    if order_list.get(user_number):
                        user_states[user_number] = "awaiting_another_dish"
                        if lang == "ar":
                            response += "\n\n✅ تمت معالجة جميع العناصر. هل ترغب في إضافة طبق آخر؟ (نعم/لا)"
                        else:
                            response += "\n\n✅ All items processed. Would you like to add another dish? "
                    else:
                        if lang == "ar":
                            response += "\n\nالرجاء محاولة طلب طبق آخر."
                        else:
                            response += "\n\nPlease try ordering another dish."
                
                # Enhance with Groq
                return enhance_response_with_groq(response, dish_str, lang)
                
            user_states[user_number + "_quantity"] = qty
            temp_order[user_number] = {"dish": dish, "quantity": qty, "customizations": None}
            user_states[user_number] = "customizing_dish"
                
            response = ""
            if lang == "ar":
                dish_name = dish.get("name_arabic", dish.get("name", ""))
                description = dish.get("description_arabic", dish.get("description", "لا يوجد وصف."))
                response = (f"🍽️ لقد اخترت {qty} × {dish_name}.\n"
                        f"الوصف: {description}\n"
                        "هل ترغب في تخصيصه؟ (أجب بـ 'نعم' أو 'لا')")
            else:
                dish_name = dish.get("name", "")
                description = dish.get("description", "No description available.")
                response = (f"🍽️ You selected {qty} x {dish_name}.\n"
                        f"Description: {description}\n"
                        "Would you like to customize it? (Reply with 'yes' or 'no')")
            
            # Enhance with Groq
            return enhance_response_with_groq(response, dish_str, lang)
        
        # Handle new multi-orders
        multi = parse_multi_order(user_input)
        if multi and len(multi) > 0:
            user_states[user_number + "_multi"] = multi
            user_states[user_number + "_multi_index"] = 0
            return process_message(user_number, "", latitude, longitude)
        
        if user_input in ["done", "finish", "انتهيت", "تم"]:
            if order_list.get(user_number):
                user_states[user_number] = "awaiting_order_type"
                response = ""
                if lang == "ar":
                    response = "✅ تمت إضافة جميع الأطباق. هل طلبك للتوصيل أم ستأتي إلى المطعم؟"
                else:
                    response = "✅ All dishes added. Is your order for delivery or will you come to the restaurant?"
                
                # Enhance with Groq
                return enhance_response_with_groq(response, user_input, lang)
            else:
                response = ""
                if lang == "ar":
                    response = "⚠️ لم تقم بإضافة أي طبق بعد. الرجاء طلب طبق أولاً."
                else:
                    response = "⚠️ You haven't added any dish yet. Please order a dish first."
                
                # Enhance with Groq
                return enhance_response_with_groq(response, user_input, lang)
        else:
            quantity = extract_quantity(user_input)
            cleaned_text = re.sub(r"(i want to order|order|please|أريد|طلب)", "", user_input, flags=re.IGNORECASE)
            dish_name_query = re.sub(r'\d+', '', cleaned_text).strip()
            dish = get_dish_by_name(dish_name_query, lang)
            if dish:
                user_states[user_number + "_quantity"] = quantity
                temp_order[user_number] = {"dish": dish, "quantity": quantity, "customizations": None}
                user_states[user_number] = "customizing_dish"
                
                response = ""
                if lang == "ar":
                    name_text = dish.get("name_arabic", dish.get("name", ""))
                    desc_text = dish.get("description_arabic", dish.get("description", "لا يوجد وصف."))
                    response = (f"🍽️ لقد اخترت {quantity} × {name_text}.\n"
                            f"الوصف: {desc_text}\n"
                            "هل ترغب في تخصيصه؟ (أجب بـ 'نعم' أو 'لا')")
                else:
                    response = (f"🍽️ You selected {quantity} x {dish.get('name', '')}.\n"
                            f"Description: {dish.get('description', 'No description available.')}\n"
                            "Would you like to customize it? (Reply with 'yes' or 'no')")
                
                # Enhance with Groq
                return enhance_response_with_groq(response, user_input, lang)
            else:
                # Use BM25 to get better search results for suggestions
                if lang == "ar":
                    suggestions = arabic_bm25.get_top_n(normalize_arabic(dish_name_query), menu, n=3)
                    suggestions = [item[0] for item in suggestions if item[1] > 0]
                else:
                    suggestions = english_bm25.get_top_n(dish_name_query, menu, n=3)
                    suggestions = [item[0] for item in suggestions if item[1] > 0]
                
                # If BM25 didn't find anything, fall back to fuzzy matching
                if not suggestions:
                    for item in menu:
                        if lang == "ar":
                            score = process.extractOne(dish_name_query, [item.get('name_arabic', item.get('name', ''))])[1]
                        else:
                            score = process.extractOne(dish_name_query, [item.get('name', '')])[1]
                        if score >= 70:
                            suggestions.append(item)
                
                if suggestions:
                    response = ""
                    if lang == "ar":
                        response = f"🤔 إليك بعض الأطباق التي قد تعجبك (الكمية: {quantity}):\n"
                        response += "\n".join([f"{i+1}. {item.get('name_arabic', item.get('name', ''))}" for i, item in enumerate(suggestions)])
                        response += "\n📌 الرجاء الرد برقم الطبق أو اسمه."
                    else:
                        response = f"🤔 Here are some dishes you might like (Quantity: {quantity}):\n"
                        response += "\n".join([f"{i+1}. {item.get('name', '')}" for i, item in enumerate(suggestions)])
                        response += "\n📌 Please reply with the dish number or name."
                    
                    user_states[user_number + "_quantity"] = quantity
                    user_states[user_number + "_dishes"] = suggestions
                    
                    # Enhance with Groq
                    return enhance_response_with_groq(response, user_input, lang)
                else:
                    response = ""
                    if lang == "ar":
                        response = "❌ عذراً، لم نجد هذا الطبق. الرجاء المحاولة مرة أخرى أو عرض القائمة."
                    else:
                        response = "❌ Sorry, we couldn't find that dish. Please try again or view the menu."
                    
                    # Enhance with Groq
                    return enhance_response_with_groq(response, user_input, lang)
    
    # --- 3. Customizing Dish ---
    elif user_states[user_number] == "customizing_dish":
        if any(word in user_input for word in ["yes", "yeah", "yep", "نعم"]):
            user_states[user_number] = "awaiting_modification"
            response = ""
            if lang == "ar":
                response = "✏️ الرجاء تقديم تفاصيل التخصيص (مثلاً، 'جبنة إضافية، بدون بصل')."
            else:
                response = "✏️ Please provide your customization details (e.g., 'extra cheese, no onions')."
            
            # Enhance with Groq
            return enhance_response_with_groq(response, user_input, lang)
        elif any(word in user_input for word in ["no", "nah", "nope", "لا"]):
            dish_info = temp_order.pop(user_number, None)
            if dish_info:
                dish_copy = dish_info["dish"].copy()
                dish_copy["quantity"] = dish_info["quantity"]
                dish_copy["customizations"] = "None"
                order_list.setdefault(user_number, []).append(dish_copy)
                
                # Check if there are more items in multi-order to process
                if user_states.get(user_number + "_multi") and user_states.get(user_number + "_multi_index", 0) < len(user_states[user_number + "_multi"]):
                    user_states[user_number] = "awaiting_dish_selection"
                    return process_message(user_number, "", latitude, longitude)
                else:
                    user_states[user_number] = "awaiting_another_dish"
                    response = ""
                    if lang == "ar":
                        response = "✅ تمت إضافة الطبق إلى طلبك. هل ترغب في إضافة طبق آخر؟ (نعم/لا)"
                    else:
                        response = "✅ Dish added to your order. Would you like to add another dish? "
                    
                    # Enhance with Groq
                    return enhance_response_with_groq(response, user_input, lang)
            else:
                response = ""
                if lang == "ar":
                    response = "⚠️ حدث خطأ. الرجاء المحاولة مرة أخرى."
                else:
                    response = "⚠️ An error occurred. Please try ordering the dish again."
                
                # Enhance with Groq
                return enhance_response_with_groq(response, user_input, lang)
        else:
            response = ""
            if lang == "ar":
                response = "❓ الرجاء الرد بـ 'نعم' أو 'لا'."
            else:
                response = "❓ Please reply with 'yes' or 'no'."
            
            # Enhance with Groq
            return enhance_response_with_groq(response, user_input, lang)
    
    # --- 4. Awaiting Modification Details ---
    elif user_states[user_number] == "awaiting_modification":
        dish_info = temp_order.pop(user_number, None)
        if dish_info:
            dish_copy = dish_info["dish"].copy()
            dish_copy["quantity"] = dish_info["quantity"]
            dish_copy["customizations"] = user_input
            order_list.setdefault(user_number, []).append(dish_copy)
            
            # Check if there are more items in multi-order to process
            if user_states.get(user_number + "_multi") and user_states.get(user_number + "_multi_index", 0) < len(user_states[user_number + "_multi"]):
                user_states[user_number] = "awaiting_dish_selection"
                return process_message(user_number, "", latitude, longitude)
            else:
                user_states[user_number] = "awaiting_another_dish"
                response = ""
                if lang == "ar":
                    response = "✅ تم حفظ التخصيص. هل ترغب في إضافة طبق آخر؟ (نعم/لا)"
                else:
                    response = "✅ Customization saved. Would you like to add another dish? "
                
                # Enhance with Groq
                return enhance_response_with_groq(response, user_input, lang)
        else:
            response = ""
            if lang == "ar":
                response = "⚠️ حدث خطأ. الرجاء المحاولة مرة أخرى."
            else:
                response = "⚠️ An error occurred. Please try ordering the dish again."
            
            # Enhance with Groq
            return enhance_response_with_groq(response, user_input, lang)
    
    # --- 5. Add Another Dish? ---
    elif user_states[user_number] == "awaiting_another_dish":
        if any(word in user_input for word in ["yes", "yep", "نعم"]):
            user_states[user_number] = "awaiting_dish_selection"
            response = ""
            if lang == "ar":
                response = ("👍 الرجاء كتابة اسم الطبق التالي الذي ترغب بطلبه. "
                        "يمكنك تحديد الكمية (مثلاً، '2 باستا' أو 'ثلاث سلطات').")
            else:
                response = ("👍 Please type the name of the next dish you'd like to order. "
                        "You can specify quantity (e.g., '2 pasta' or 'three salads').")
            
            # Enhance with Groq
            return enhance_response_with_groq(response, user_input, lang)
        elif any(word in user_input for word in ["no", "nah", "لا"]):
            if order_list.get(user_number):
                user_states[user_number] = "awaiting_order_type"
                response = ""
                if lang == "ar":
                    response = "✅ تمت إضافة جميع الأطباق. هل طلبك للتوصيل أم ستأتي إلى المطعم؟"
                else:
                    response = "✅ All dishes added. Is your order for delivery or will you come to the restaurant?"
                
                # Enhance with Groq
                return enhance_response_with_groq(response, user_input, lang)
            else:
                response = ""
                if lang == "ar":
                    response = "⚠️ طلبك فارغ. الرجاء إضافة طبق واحد على الأقل."
                else:
                    response = "⚠️ Your order is empty. Please add at least one dish."
                
                # Enhance with Groq
                return enhance_response_with_groq(response, user_input, lang)
        else:
            response = ""
            if lang == "ar":
                response = "❓ الرجاء الرد بـ 'نعم' أو 'لا'."
            else:
                response = "❓ Please reply with 'yes' or 'no'."
            
            # Enhance with Groq
            return enhance_response_with_groq(response, user_input, lang)
    
    # --- 6. Awaiting Order Type (Delivery or Dine-in) ---
    elif user_states[user_number] == "awaiting_order_type":
        if "delivery" in user_input or "توصيل" in user_input:
            user_states[user_number + "_order_type"] = "delivery"
            user_states[user_number] = "awaiting_restaurant_branch"
            response = ""
            if lang == "ar":
                response = ("📍 من أي فرع من فروع الفرّوج ترغب في الطلب؟ الرجاء الاختيار من: مزمار, عجمان, الشارقة, توام, المناصير")
            else:
                response = ("📍 Which Al Farooj branch would you like to order from? Please choose from: Mazmar, Ajman, Sharjah, Tawam, Manaseer.")
            
            # Enhance with Groq
            return enhance_response_with_groq(response, user_input, lang)
        elif any(word in user_input for word in ["dine", "restaurant", "come", "pickup", "تناول", "مطعم"]):
            user_states[user_number + "_order_type"] = "dine-in"
            user_states[user_number] = "awaiting_restaurant_branch"
            response = ""
            if lang == "ar":
                response = ("📍 أي فرع من فروع الفرّوج ستزوره؟ الرجاء الاختيار من: مزمار, عجمان, الشارقة, توام, المناصير.")
            else:
                response = ("📍 Which Al Farooj branch will you visit? Please choose from: Mazmar, Ajman, Sharjah, Tawam, Manaseer.")
            
            # Enhance with Groq
            return enhance_response_with_groq(response, user_input, lang)
        else:
            response = ""
            if lang == "ar":
                response = "❓ الرجاء تحديد إذا كان طلبك 'توصيل' أو 'تناول في المطعم'."
            else:
                response = "❓ Please specify if your order is 'delivery' or 'dine-in'."
            
            # Enhance with Groq
            return enhance_response_with_groq(response, user_input, lang)
    
    # --- 7. Awaiting Restaurant Branch Selection ---
    elif user_states[user_number] == "awaiting_restaurant_branch":
        branch = user_input.strip().title()
        valid_branches = ["Mazmar", "Ajman", "Sharjah", "Tawam", "Manaseer"]
        arabic_branches = ["مزمار", "عجمان", "الشارقة", "توام", "المناصير"]
        branch_mapping = dict(zip(arabic_branches, valid_branches))
        reverse_branch_mapping = dict(zip(valid_branches, arabic_branches))
        if branch in arabic_branches:
            branch = branch_mapping[branch]
        if branch in valid_branches:
            user_states[user_number + "_branch"] = branch
            user_states[user_number] = "awaiting_payment_method" if user_states.get(user_number + "_order_type") == "delivery" else "awaiting_name"
            response = ""
            if lang == "ar":
                arabic_branch_name = reverse_branch_mapping[branch]
                if user_states[user_number] == "awaiting_payment_method":
                    response = "💳 الرجاء اختيار طريقة الدفع: نقداً أو ببطاقة."
                else:
                    response = f"✅ تم تأكيد الطلب داخل المطعم في فرع {arabic_branch_name}. الرجاء تقديم اسمك لإتمام الطلب."
            else:
                if user_states[user_number] == "awaiting_payment_method":
                    response = "💳 Please choose your payment method: cash or card."
                else:
                    response = f"✅ Dine-in order confirmed at {branch}. Please provide your name to finalize your order."
            
            # Enhance with Groq
            return enhance_response_with_groq(response, user_input, lang)
        else:
            response = ""
            if lang == "ar":
                response = f"❌ فرع غير صالح. الرجاء الاختيار من: {', '.join(arabic_branches)}."
            else:
                response = f"❌ Invalid branch. Please choose from: {', '.join(valid_branches)}."
            
            # Enhance with Groq
            return enhance_response_with_groq(response, user_input, lang)
    
    # --- 8. Awaiting Payment Method (For Delivery Only) ---
    elif user_states[user_number] == "awaiting_payment_method":
        if "cash" in user_input or "نقد" in user_input:
            user_states[user_number + "_payment"] = "cash"
            user_states[user_number] = "awaiting_delivery_location"
            response = ""
            if lang == "ar":
                response = "💵 تم حفظ طريقة الدفع كـ نقداً. الرجاء مشاركة موقع التوصيل أو كتابة عنوانك."
            else:
                response = "💵 Payment method saved as cash. Please share your delivery location or type your address."
            
            # Enhance with Groq
            return enhance_response_with_groq(response, user_input, lang)
        elif "card" in user_input or "بطاقة" in user_input:
            user_states[user_number + "_payment"] = "card"
            user_states[user_number] = "awaiting_delivery_location"
            response = ""
            if lang == "ar":
                response = "💳 تم حفظ طريقة الدفع كبطاقة. الرجاء مشاركة موقع التوصيل أو كتابة عنوانك."
            else:
                response = "💳 Payment method saved as card. Please share your delivery location or type your address."
            
            # Enhance with Groq
            return enhance_response_with_groq(response, user_input, lang)
        else:
            response = ""
            if lang == "ar":
                response = "❌ طريقة دفع غير صالحة. الرجاء كتابة 'نقد' أو 'بطاقة'."
            else:
                response = "❌ Invalid payment method. Please type 'cash' or 'card'."
            
            # Enhance with Groq
            return enhance_response_with_groq(response, user_input, lang)
    
    # --- 9. Awaiting Delivery Location (For Delivery Only) ---
    elif user_states[user_number] == "awaiting_delivery_location":
        if latitude and longitude:
            user_states[user_number + "_location"] = f"📍 Lat: {latitude}, Lng: {longitude}"
        elif user_input:
            user_states[user_number + "_location"] = user_input
        else:
            response = ""
            if lang == "ar":
                response = "❓ الرجاء مشاركة موقع التوصيل باستخدام زر الإرفاق أو كتابة عنوانك."
            else:
                response = "❓ Please share your delivery location using the attach button or type your address."
            
            # Enhance with Groq
            return enhance_response_with_groq(response, user_input, lang)
        
        user_states[user_number] = "awaiting_name"
        response = ""
        if lang == "ar":
            response = "📌 تم استلام موقع التوصيل. الرجاء تقديم اسمك لإتمام الطلب."
        else:
            response = "📌 Delivery location received. Please provide your name to finalize your order."
        
        # Enhance with Groq
        return enhance_response_with_groq(response, user_input, lang)
    
    # --- 10. Awaiting Name (Finalize Order) ---
    elif user_states[user_number] == "awaiting_name":
        user_name = user_input.strip().title()
        if not user_name:
            response = ""
            if lang == "ar":
                response = "⚠️ الرجاء إدخال اسم صالح."
            else:
                response = "⚠️ Please enter a valid name."
            
            # Enhance with Groq
            return enhance_response_with_groq(response, user_input, lang)
        
        aggregated = {}
        for item in order_list.get(user_number, []):
            if lang == "ar":
                dish_name = item.get("name_arabic", item.get("name", "طبق غير معروف"))
            else:
                dish_name = item.get("name", "Unknown Dish")
            price = item.get("price", 0)
            qty = item.get("quantity", 1)
            key = (dish_name, price)
            aggregated[key] = aggregated.get(key, 0) + qty
        
        total = 0
        invoice = "🧾 Invoice:\n" if lang != "ar" else "🧾 الفاتورة:\n"
        for (name, price), qty in aggregated.items():
            line_total = price * qty
            total += line_total
            if lang == "ar":
                invoice += f"{qty} × {name} بسعر {price:.2f} درهم = {line_total:.2f} درهم\n"
            else:
                invoice += f"{qty} x {name} @ AED {price:.2f} = AED {line_total:.2f}\n"
        invoice += f"Total: AED {total:.2f}" if lang != "ar" else f"الإجمالي: {total:.2f} درهم"
        
        order_id = str(uuid.uuid4())[:8]
        order_type = user_states.get(user_number + "_order_type", "dine-in")
        branch = user_states.get(user_number + "_branch", "Not specified")
        payment_method = user_states.get(user_number + "_payment", "N/A")
        user_location = user_states.get(user_number + "_location", "N/A")
        
        final_order = {
            "order_id": order_id,
            "user_id": user_number,
            "username": user_name,
            "order_type": order_type,
            "branch": branch,
            "location": user_location,
            "payment_method": payment_method,
            "order": order_list.get(user_number, []),
            "total_cost": total,
            "timestamp": datetime.now().isoformat()
        }
        
        finalize_order_to_json(final_order)
        reset_server()
        
        response = ""
        if lang == "ar":
            response = (f"✅ شكراً لك {user_name}! تم تأكيد طلبك (المعرف: {order_id}).\n"
                    f"{invoice}\nاستمتع بوجبتك! 🍽️")
        else:
            response = (f"✅ Thank you {user_name}! Your order (ID: {order_id}) has been confirmed.\n"
                    f"{invoice}\nEnjoy your meal! 🍽️")
        
        # Enhance with Groq
        return response
    
    # --- 11. Final Confirmation or Cancellation (if applicable) ---
    elif user_states[user_number] == "awaiting_final_confirmation":
        if "confirm" in user_input or "تأكيد" in user_input or "yes" in user_input:
            user_states[user_number] = "awaiting_name"
            response = ""
            if lang == "ar":
                response = "📝 الرجاء إدخال اسمك لتأكيد الطلب:"
            else:
                response = "📝 Please enter your name for the order confirmation:"
            
            # Enhance with Groq
            return enhance_response_with_groq(response, user_input, lang)
        elif "cancel" in user_input or "إلغاء" in user_input:
            reset_server()
            response = ""
            if lang == "ar":
                response = "❌ تم إلغاء طلبك. تم إعادة تعيين الجلسة."
            else:
                response = "❌ Your order has been canceled. The session has been reset."
            
            # Enhance with Groq
            return enhance_response_with_groq(response, user_input, lang)
        else:
            response = ""
            if lang == "ar":
                response = "❓ الرجاء الرد بـ 'تأكيد' أو 'إلغاء'."
            else:
                response = "❓ Please respond with 'confirm' or 'cancel'."
            
            # Enhance with Groq
            return enhance_response_with_groq(response, user_input, lang)
    
    else:
        response = ""
        if lang == "ar":
            response = "🤖 عذراً، لم أفهم ذلك. الرجاء المحاولة مرة أخرى."
        else:
            response = "🤖 Sorry, I didn't understand that. Please try again."
        
        # Enhance with Groq
        return enhance_response_with_groq(response, user_input, lang)

# ---------------------------
# Logging Function
# ---------------------------
LOG_FILE = "whatsapp_logs.json"
def log_message(user_number, incoming_msg, reply):
    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user_number": user_number,
        "incoming_message": incoming_msg,
        "response": reply if isinstance(reply, str) else repr(reply)
    }
    
    # Make sure log directory exists
    if not os.path.exists(os.path.dirname(LOG_FILE)) and os.path.dirname(LOG_FILE):
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r", encoding="utf-8") as file:
            try:
                logs = json.load(file)
            except json.JSONDecodeError:
                logs = []
    else:
        logs = []
    
    logs.append(log_entry)
    with open(LOG_FILE, "w", encoding="utf-8") as file:
        json.dump(logs, file, indent=4, ensure_ascii=False)

# ---------------------------
# Flask Routes for UI Chat
# ---------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/message", methods=["POST"])
def message():
    data = request.get_json()
    user_input = data.get("message", "")
    latitude = data.get("latitude")
    longitude = data.get("longitude")
    
    # Using a fixed user number (or you could use session management)
    user_number = "user"
    reply = process_message(user_number, user_input, latitude, longitude)
    log_message(user_number, user_input, reply)
    
    # If reply includes images (as in the menu), return both text and images.
    if isinstance(reply, dict) and "menu_images" in reply:
        return jsonify({
            "text": reply.get("text", ""),
            "menu_images": reply.get("menu_images", [])
        })
    else:
        return jsonify({"text": reply})

if __name__ == "__main__":
    app.run(debug=True)