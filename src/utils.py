import numpy as np

COLORS = ["red", "blue", "yellow", "green", "black", "white", "purple", "pink"]
TYPES = ["electric", "fire", "water", "grass", "psychic", "dragon", "ice", "dark"]
OBJECTS = ["pokemon", "creature", "character"]

def get_predicted_keywords(model, X_sample, mlb, max_words=5):
    X_sample = X_sample.reshape(1, -1)
    y_pred = model.predict(X_sample)
    keywords = mlb.inverse_transform(y_pred)
    return list(keywords[0])[:max_words]

def generate_caption(keywords):
    """Template-based caption generator."""
    if not keywords:
        return "A Pokémon character."
        
    # Attempt specific template matching
    color = next((w for w in keywords if w in COLORS), None)
    ptype = next((w for w in keywords if w in TYPES), None)
    obj = next((w for w in keywords if w in OBJECTS), "pokemon")

    if color and ptype:
        return f"This image shows a {color} {ptype} {obj}."
    elif color:
        return f"This image shows a {color} {obj}."
    elif ptype:
        return f"This image shows a {ptype} {obj}."
    
    # Fallback generic generation
    sentence = "A Pokémon with " + ", ".join(keywords[:-1])
    if len(keywords) > 1:
        sentence += " and " + keywords[-1]
    else:
        sentence = f"A Pokémon with {keywords[0]}"
    return sentence + "."