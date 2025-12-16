def text_to_pad(text):
    if "excited" in text.lower():
        return (0.9, 0.8, 0.6)
    elif "sad" in text.lower():
        return (0.2, 0.3, 0.4)
    else:
        return (0.5, 0.5, 0.5)