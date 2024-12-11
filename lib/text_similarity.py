from evaluate import load

def text_similarity(text1, text2):
    bertscore_model = load("bertscore")
    return bertscore_model.compute(text1, text2)
