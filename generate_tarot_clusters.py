import gensim.downloader as api
from gensim.models import KeyedVectors
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet as wn

# Download required NLTK resources
nltk.download("averaged_perceptron_tagger")
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# Filter only en/* words and save model, only need to do this once
print("Loading ConceptNet Numberbatch vectors...")
model = api.load("conceptnet-numberbatch-17-06-300")  # ~500MB
en_words = [w for w in model.index_to_key if w.startswith("/c/en/")]
en_model = model.__class__(vector_size=model.vector_size)
en_model.add_vectors(en_words, [model[w] for w in en_words])
en_model.save("numberbatch-en-only.kv")

# In subsequent runs, load model this way
#model = KeyedVectors.load("numberbatch-en-only.kv", mmap='r') 

# Step 1: Filter down to lowercase, alphabetic, non-stopword, non-blacklist
all_words = list(model.index_to_key)
stop_words = set(stopwords.words("english"))

def is_common_noun(word):
    synsets = wn.synsets(word, pos=wn.NOUN)
    for s in synsets:
        if 'noun.person' in s.lexname() or 'noun.location' in s.lexname():
            return False
    return bool(synsets)

def extract_concept_label(uri):
    # Only keep concepts like /c/en/word (ignore multiword for now)
    parts = uri.split("/")
    if len(parts) >= 4 and parts[2] == "en":
        label = parts[3]
        if "_" in label:
            return None  # skip multiword for now
        return label
    return None

filtered_words = []
chunk_size = 1000
print("Filtering down to lowercase, non-proper common nouns...")
for i in range(0, len(all_words), chunk_size):
    chunk = all_words[i:i + chunk_size]
    
    for full_uri in chunk:
        label = extract_concept_label(full_uri)
        if label is None:
            continue
        if not label.isalpha() or label in stop_words:
            continue
        tag = pos_tag([label])[0][1]
        if tag == "NN" and is_common_noun(label):
            filtered_words.append(full_uri)  # keep the full /c/en/word form
        if len(filtered_words) >= 100000:
            break
    if len(filtered_words) >= 100000:
        break

# Step 2: Semantic similarity to Major Arcana seed words
seed_words = [
    "love"
]
seed_words_prefixed = [f"/c/en/{word}" for word in seed_words]
available_seed_words = [word for word in seed_words_prefixed if word in model]
if not available_seed_words:
    raise ValueError("None of the seed words are in the embedding model!")

seed_vectors = np.array([model[word] for word in available_seed_words])

print(f"Using {len(available_seed_words)} seed words: {', '.join(available_seed_words)}")

word_vectors = np.array([model[w] for w in filtered_words])

similarities = cosine_similarity(word_vectors, seed_vectors)
avg_similarity = similarities.mean(axis=1)

# Step 3: Keep top 1000 semantically closest
top_indices = np.argsort(avg_similarity)[-1000:]
words = [filtered_words[i] for i in top_indices]
vectors = word_vectors[top_indices]

# Step 4: KMeans clustering
print("Clustering into 22 concept groups...")
kmeans = KMeans(n_clusters=22, random_state=42, n_init="auto")
kmeans.fit(vectors)

# Step 5: Closest 10 words per cluster
cluster_keywords = []
for i in range(22):
    center = kmeans.cluster_centers_[i]
    cluster_indices = np.where(kmeans.labels_ == i)[0]
    cluster_words = [words[j] for j in cluster_indices]
    cluster_vectors = vectors[cluster_indices]
    distances = np.linalg.norm(cluster_vectors - center, axis=1)
    sorted_indices = np.argsort(distances)
    closest_words = [cluster_words[j] for j in sorted_indices[:10]]
    cluster_keywords.append(closest_words)

# Step 6: Display results
print("\n🃏 Tarot-like Semantic Clusters (ConceptNet):")
for i, word_list in enumerate(cluster_keywords, 1):
    main_word = word_list[0]
    secondary_words = ", ".join(word_list[1:])
    print(f"{i:2d}. {main_word} (→ {secondary_words})")