"""
Disease cluster extraction from radiology captions.
Creates disease vectors for cluster-aware contrastive learning.
"""
import re
from typing import List, Dict, Set, Tuple
import config


def extract_diseases(caption: str, keywords: Dict[str, List[str]] = None) -> Set[str]:
    """Extract disease categories from a radiology caption."""
    if keywords is None:
        keywords = config.DISEASE_KEYWORDS
    
    caption_lower = caption.lower()
    diseases = set()
    
    for category, terms in keywords.items():
        for term in terms:
            if term.lower() in caption_lower:
                diseases.add(category)
                break
    
    if not diseases:
        diseases.add('other')
    
    return diseases


def build_disease_matrix(captions: List[str]) -> Tuple[List[Set[str]], Dict[str, int]]:
    """
    Build disease sets for all captions and create category-to-index mapping.
    Returns: (list of disease sets, category name -> index dict)
    """
    all_categories = set()
    disease_sets = []
    
    for caption in captions:
        diseases = extract_diseases(caption)
        disease_sets.append(diseases)
        all_categories.update(diseases)
    
    category_to_idx = {cat: idx for idx, cat in enumerate(sorted(all_categories))}
    return disease_sets, category_to_idx


def compute_disease_similarity(
    diseases_i: Set[str],
    diseases_j: Set[str],
    temperature: float = 0.5
) -> float:
    """
    Compute disease similarity between two samples.
    Uses Jaccard similarity with exponential scaling.
    """
    if diseases_i & diseases_j:  # Share at least one disease
        intersection = len(diseases_i & diseases_j)
        union = len(diseases_i | diseases_j)
        jaccard = intersection / union if union > 0 else 0.0
        return jaccard
    return 0.0


def build_similarity_matrix(
    disease_sets: List[Set[str]],
    temperature: float = 0.5
) -> List[List[float]]:
    """Build full pairwise disease similarity matrix."""
    n = len(disease_sets)
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            matrix[i][j] = compute_disease_similarity(
                disease_sets[i], disease_sets[j], temperature
            )
    return matrix


def build_similarity_matrix_batch(
    disease_sets_batch: List[Set[str]],
    all_disease_sets: List[Set[str]],
    temperature: float = 0.5
) -> List[List[float]]:
    """
    Build similarity matrix between batch samples and all samples.
    More memory efficient for large datasets.
    """
    n_batch = len(disease_sets_batch)
    n_all = len(all_disease_sets)
    matrix = [[0.0] * n_all for _ in range(n_batch)]
    for i in range(n_batch):
        for j in range(n_all):
            matrix[i][j] = compute_disease_similarity(
                disease_sets_batch[i], all_disease_sets[j], temperature
            )
    return matrix


if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv(config.CSV_PATH)
    disease_sets, cat2idx = build_disease_matrix(df['org_caption'].tolist())
    
    print(f"Total samples: {len(disease_sets)}")
    print(f"Disease categories: {cat2idx}")
    print(f"\nCategory distribution:")
    from collections import Counter
    all_cats = [cat for ds in disease_sets for cat in ds]
    for cat, count in Counter(all_cats).most_common():
        print(f"  {cat}: {count} ({count/len(disease_sets)*100:.1f}%)")
