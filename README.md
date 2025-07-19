# shuangxi
Generates tarot arcana by clustering word embeddings associated with love

# Double Happiness Tarot

Generates 22 semantically rich tarot arcana based on the theme of love by clustering ConceptNet Numberbatch word embeddings. 

## How it Works

1. **Load ConceptNet Numberbatch**: Uses pre-trained word embeddings to represent concepts as vectors.
2. **Filter Concepts**: Keeps only lowercase English common nouns, excluding stopwords and proper nouns.
3. **Semantic Closeness to Love**: Ranks concepts by cosine similarity to the seed word "love".
4. **KMeans Clustering**: Clusters the top 1,000 love-adjacent words into 22 semantic groups, mirroring the Major Arcana.
5. **Labeling Each Card**: Each group is represented by its closest 10 words, serving as the symbolic meanings for each card.

## Installation

```bash
git clone https://github.com/yourusername/double-happiness-tarot.git
cd double-happiness-tarot
pip install -r requirements.txt