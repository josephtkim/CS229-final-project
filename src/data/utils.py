import re

def remove_punc_special(text):
    return ''.join(char.lower() for char in text if char.isalpha() or char.isspace())

def clean_and_validate_attributes(text_list):
    valid_attributes = {
        'young', 'male', 'female', 'smiling', 'eyeglasses',
        'black hair', 'blond hair', 'bald', 'mustache', 'wearing lipstick'
    }
    cleaned_words = []
    for text in text_list:
        cleaned_text = remove_punc_special(text)
        cleaned_words.extend(cleaned_text.split())

    found_attributes = []
    i = 0
    while i < len(cleaned_words):
        if i < len(cleaned_words)-1:
            two_words = f"{cleaned_words[i]} {cleaned_words[i+1]}"
            if two_words in valid_attributes:
                found_attributes.append(two_words)
                i+=2
                continue
        if cleaned_words[i] in valid_attributes:
            found_attributes.append(cleaned_words[i])
        i+=1
    return found_attributes

def generate_natural_description(text):
    attributes = clean_and_validate_attributes(text) if not isinstance(text, list) else clean_and_validate_attributes(text)
    if not attributes:
        return "A person."

    unique_attributes = set(attributes)
    parts = ['a']

    if 'young' in unique_attributes:
        parts.append('young')
        unique_attributes.remove('young')

    if 'male' in unique_attributes and 'female' in unique_attributes:
        parts.append('male')
        unique_attributes.discard('male')
        unique_attributes.discard('female')
    elif 'male' in unique_attributes:
        parts.append('male')
        unique_attributes.discard('male')
    elif 'female' in unique_attributes:
        parts.append('female')
        unique_attributes.discard('female')
    else:
        parts.append('person')

    special_attrs = ['smiling', 'bald', 'wearing lipstick', 'wearing hat']
    special_parts = []
    for attr in special_attrs:
        if attr in unique_attributes:
            special_parts.append(attr)
            unique_attributes.remove(attr)

    if special_parts:
        parts.append('who is ' + ' and '.join(special_parts))

    if unique_attributes:
        parts.append('with')
        parts.append(' and '.join(unique_attributes))

    return ' '.join(parts) + '.'

def token_f1_loss(ground_truth, reconstructed):
    gt_tokens = set(clean_and_validate_attributes(ground_truth.split()))
    recon_tokens = set(clean_and_validate_attributes(reconstructed.split()))

    key_attrs = {'male', 'female', 'young'}
    key_weight = 2.0

    weighted_tp = sum(key_weight if t in key_attrs else 1.0 for t in gt_tokens & recon_tokens)
    weighted_fp = sum(key_weight if t in key_attrs else 1.0 for t in recon_tokens - gt_tokens)
    weighted_fn = sum(key_weight if t in key_attrs else 1.0 for t in gt_tokens - recon_tokens)

    precision = weighted_tp / (weighted_tp + weighted_fp) if (weighted_tp + weighted_fp)>0 else 0
    recall = weighted_tp / (weighted_tp + weighted_fn) if (weighted_tp + weighted_fn)>0 else 0
    f1_score = (2*precision*recall)/(precision+recall) if (precision+recall)>0 else 0
    return 1 - f1_score**2

def attribute_loss(ground_truth, reconstructed):
    gt_attributes = clean_and_validate_attributes(ground_truth.split())
    recon_attributes = clean_and_validate_attributes(reconstructed.split())
    intersection = len(set(gt_attributes) & set(recon_attributes))
    union = len(set(gt_attributes) | set(recon_attributes))
    jaccard_similarity = intersection / union if union>0 else 0.0
    return 1 - jaccard_similarity
    